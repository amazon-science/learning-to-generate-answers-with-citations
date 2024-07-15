import argparse
import collections
import copy
import json
import logging
import re
import string
import sys

import numpy as np
import scipy as sp
import shap
import torch
from nltk import sent_tokenize
from tqdm import tqdm

from src.evaluation.alignscore import AlignScore

logger = logging.getLogger(__name__)


from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.utils.util import get_max_memory, normalize_answer, remove_citations


def get_bert_score(references, candidates):
    """
    Taken from https://github.com/deng1fan/AgileLightning
    """
    from bert_score import score

    scores = score(
        candidates,
        references,
        lang="en",
        verbose=False,
        rescale_with_baseline=True,
        device="cuda",
    )[-1].numpy()
    return round(np.mean(list(scores)), 4)


def compute_len(data):
    """Compute average length of predictions."""

    res, cntr = 0, 0
    for item in data:
        res += len(item["output"].split())
        cntr += 1
    return res / cntr


def compute_qa(qa_pairs, generations, qa_model="gaotianyu1350/roberta-large-squad"):
    """Compute QA-based accuracy.
    Args:
        data: requires filed `qa_pairs/short_answers` and `output`
    Returns:
        QA metrics (QA-EM, QA-F1, QA-Hit)
    """

    def compute_exact(a_gold, a_pred):
        """Check whether two strings are equal up to normalization."""

        return int(normalize_answer(a_gold) == normalize_answer(a_pred))

    def compute_f1(a_gold, a_pred):
        """Compute F1 score between two strings."""

        def _get_tokens(s):
            if not s:
                return []
            return normalize_answer(s).split()

        gold_toks = _get_tokens(a_gold)
        pred_toks = _get_tokens(a_pred)

        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())

        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)

        if num_same == 0:
            return 0

        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1

    if qa_pairs is None:
        logger.warn("Warning: no QA pairs found in data")
        return {
            "QA-EM": 0,
            "QA-F1": 0,
            "QA-Hit": 0,
        }

    # Load model
    logger.info("Loading the RoBERTa-large SQuAD model for QA-based accuracy...")
    qa_pipeline = pipeline("question-answering", model=qa_model, device=0)
    logger.info("Done")

    # Get prediction
    logger.info("Computing the QA-based accuracy...")
    em, f1, bins = [], [], []
    for i, qa_pair in tqdm(enumerate(qa_pairs)):
        questions = [ele["question"] for ele in qa_pair]
        context = generations[i] if len(generations[i]) > 0 else " "
        results = qa_pipeline(question=questions, context=context, handle_impossible_answer=True)
        loc_counter, loc_em, loc_f1 = 0, 0, 0

        for idx, res in enumerate(results):
            answers = qa_pair[idx]["short_answers"]
            prediction = res["answer"]

            loc_em += max([compute_exact(a, prediction) for a in answers])
            loc_f1 += max([compute_f1(a, prediction) for a in answers])
            loc_counter += 1

        em.append(loc_em / loc_counter)
        f1.append(loc_f1 / loc_counter)
        bins.append(loc_em == loc_counter)

    return {"QA-EM": 100 * np.mean(em), "QA-F1": 100 * np.mean(f1), "QA-Hit": 100 * np.mean(bins)}


def compute_mauve(questions, answers, generations, device_id):
    """Compute Mauve score."""

    logger.info("Computing MAUVE...")
    # generations = [remove_citations(x) for x in generations]

    human_data = []
    model_data = []
    for i in range(len(questions)):
        # Remove ending punctuations
        # Remove any new lines
        # Truncate by 100 words, following the ALCE paper
        human_data.append(" ".join((questions[i] + " " + answers[i].strip()).split()[:100]).rstrip(string.punctuation))
        model_data.append(
            " ".join((questions[i] + " " + generations[i].strip()).split()[:100]).rstrip(string.punctuation)
        )

        # human_data.append(' '.join((questions[i] + " " + answers[i].strip()).split()).rstrip(string.punctuation))
        # model_data.append(' '.join((questions[i] + " " + generations[i].strip()).split()).rstrip(string.punctuation))

    import mauve

    out = mauve.compute_mauve(
        p_text=human_data,
        q_text=model_data,
        device_id=torch.cuda.device_count() - (device_id + 1),
        max_text_length=512,
        verbose=True,
        batch_size=8,
        featurize_model_name="gpt2-large",
    )

    score = out.mauve * 100
    del out

    return score


class AutoAIS:
    def __init__(
        self,
        model_name="google/t5_xxl_true_nli_mixture",
        threshold=0.7,
        rank=0,
        compute_shap=False,
        shap_normalization="min_max",
    ):
        self.shap_normalization = shap_normalization
        self.model_name = model_name
        if "t5" in model_name:
            self.autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                max_memory=get_max_memory(rank=rank),
                device_map="auto",
                offload_folder="offload/autoais_{}".format(rank),
            )
            self.autoais_tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            print("AUTOAIS DEVICE", self.autoais_model.device)
            self.autoais_model.eval()
        elif "alignscore" in model_name:
            """
            https://arxiv.org/pdf/2305.16739.pdf
            """
            self.autoais_model = AlignScore(
                model="roberta-large",
                batch_size=32,
                device="cuda:{}".format(torch.cuda.device_count() - (rank + 1)),
                ckpt_path="models/AlignScore-large.ckpt",
                evaluation_mode="bin_sp",
            )
            self.autoais_tokenizer = self.autoais_model.model.tokenizer
            self.threshold = threshold
            # self.autoais_model.model.model.eval() # Called already in alignscore.py
        elif "little_push" in model_name:
            """
            Taken from https://github.com/julmaxi/with_a_little_push/blob/master/nlifactspush/score.py
            https://arxiv.org/pdf/2305.16819.pdf
            """
            self.autoais_model = AutoModelForSequenceClassification.from_pretrained(
                "juliussteen/DeBERTa-v3-FaithAug"
            ).to("cuda:{}".format(torch.cuda.device_count() - (rank + 1)))
            self.autoais_tokenizer = AutoTokenizer.from_pretrained("juliussteen/DeBERTa-v3-FaithAug", use_fast=False)
            self.threshold = threshold
            self.autoais_model.eval()
        else:
            logger.error("AutoAIS model with name {} not recognized, abort...".format(model_name))

        if compute_shap:
            if "alignscore" in model_name:

                def f(x):
                    outputs = []
                    for _x in x:
                        encoding = self.autoais_tokenizer(
                            _x,
                            truncation="only_first",
                            padding="max_length",
                            max_length=self.autoais_tokenizer.model_max_length,
                            return_tensors="pt",
                        ).to(self.autoais_model.model.device)
                        output = (
                            self.autoais_model.model.softmax(
                                self.autoais_model.model.model(encoding).seq_relationship_logits
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        outputs.append(output[0])
                    outputs = np.array(outputs)
                    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
                    val = sp.special.logit(scores)
                    return val

                labels = ["inconsistent", "consistent"]
                self.explainer = shap.Explainer(f, self.autoais_tokenizer, output_names=labels)
            else:
                logger.error(
                    "AutoAIS model with name {} does not have shap computation implemented, abort...".format(
                        model_name
                    )
                )

    def compute_claims_normalized(self, claims_all, generations, attributed_sentences):
        logger.info("Computing claims normalized...")
        scores = []
        for i in tqdm(range(len(claims_all))):
            if len(claims_all[i]) == 0:
                continue
            normalized_output = remove_citations(generations[i])
            normalized_output = sent_tokenize(normalized_output)
            normalized_output = "".join([x for j, x in enumerate(normalized_output) if j in attributed_sentences[i]])
            entail = 0
            claims = claims_all[i]
            for claim in claims:
                entail += self._run_nli_autoais(passage=normalized_output, claim=claim)
            scores.append(entail / len(claims))
        if scores:
            scores = 100 * np.mean(scores)
        else:
            scores = 0
        return scores

    def compute_claims(self, claims_all, generations):
        logger.info("Computing claims...")
        scores = []
        for i in tqdm(range(len(claims_all))):
            if len(claims_all[i]) == 0:
                continue
            normalized_output = remove_citations(generations[i])
            entail = 0
            claims = claims_all[i]
            for claim in claims:
                entail += self._run_nli_autoais(passage=normalized_output, claim=claim)
            scores.append(entail / len(claims))
        if scores:
            scores = 100 * np.mean(scores)
        else:
            scores = 0
        return scores

    @torch.no_grad()
    def _run_nli_autoais(self, passage, claim):
        """
        Run inference for assessing AIS between a premise and hypothesis.
        Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
        """
        if "t5" in self.model_name:
            input_text = "premise: {} hypothesis: {}".format(passage, claim)
            input_ids = self.autoais_tokenizer(
                input_text, return_tensors="pt", truncation=True, max_length=2048
            ).input_ids.to(self.autoais_model.device)
            # print(input_ids)
            with torch.inference_mode():
                outputs = self.autoais_model.generate(input_ids, max_new_tokens=5)
            result = self.autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
            inference = 1 if result == "1" else 0
            del result
            del input_ids
        elif "alignscore" in self.model_name:
            score = self.autoais_model.score(contexts=[passage], claims=[claim])[0]
            if score > self.threshold:
                inference = 1
            else:
                inference = 0
        elif "little_push" in self.model_name:
            tokens = self.autoais_tokenizer(
                passage, claim, truncation="only_first", max_length=512, return_tensors="pt"
            )
            predictions = self.autoais_model(
                input_ids=tokens.input_ids.to(self.autoais_model.device),
                attention_mask=tokens.attention_mask.to(self.autoais_model.device),
            ).logits
            score = predictions[:, 0] - predictions[:, 2]  # difference between entailed and refuted score
            if score > 0:
                inference = 1
            else:
                inference = 0

        return inference

    def _format_document(self, doc):
        """Format document for AutoAIS."""

        if "sent" in doc:
            # QA-extracted docs
            return "Title: %s\n%s" % (doc["title"], doc["sent"])
        else:
            return "Title: %s\n%s" % (doc["title"], doc["text"])

    @torch.no_grad()
    def compute_attributed_sentences(self, questions, generations, passages):
        """
        Computes whether a generated senteces is attributed in retrieved passages. Ignores whether generated sentences attributes correctly.
        """
        all_attributed_sentences = []
        for i in tqdm(range(len(questions))):
            attributed_sentences = []
            generations[i] = generations[i].replace(
                "</s>", " "
            )  # sometimes generates sentence end marker in output without ending?
            sents = sent_tokenize(generations[i])

            if len(sents) == 0:
                all_attributed_sentences.append([])
                continue

            target_sents = [remove_citations(sent).strip() for sent in sents]

            for sent_id, sent in enumerate(sents):
                if len(target_sents[sent_id].strip()) == 0:
                    continue

                target_sent = target_sents[sent_id]  # Citation removed and (if opted for) decontextualized
                joint_entail = -1  # Undecided

                joint_passage = "\n".join([self._format_document(passages[i][j]) for j in range(len(passages[i]))])

                joint_entail = self._run_nli_autoais(joint_passage, target_sent)
                if joint_entail:
                    attributed_sentences.append(sent_id)

            all_attributed_sentences.append(attributed_sentences)

        return all_attributed_sentences

    @torch.no_grad()
    def compute_autoais(
        self,
        questions,
        generations,
        passages,
        decontext=False,
        concat=False,
        qampari=False,
        at_most_citations=None,
    ):
        """
        Compute AutoAIS score.

        Args:
            data: requires field `output` and `docs`
                - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
            citation: check citations and use the corresponding references.
            decontext: decontextualize the output
        """

        logger.info(f"Running AutoAIS...")

        ais_scores = []
        ais_scores_prec = []

        ais_scores_all = []
        ais_scores_prec_all = []

        ais_scores_by_sentence = {}
        ais_scores_prec_by_sentence = {}

        sent_total = 0
        sent_mcite = 0
        sent_mcite_support = 0
        sent_mcite_overcite = 0
        autoais_log = []
        for i in tqdm(range(len(questions))):
            # Get sentences by using NLTK
            if qampari:
                sents = [
                    questions[i] + " " + x.strip()
                    for x in generations[i]
                    .rstrip()
                    .rstrip(".")
                    .rstrip(",")
                    .rstrip(QAMPARI_SEP_TOKEN)
                    .split(QAMPARI_SEP_TOKEN)
                ]
            else:

                generations[i] = generations[i].replace(
                    "</s>", " "
                )  # sometimes generates sentence end marker in output without ending?
                sents = sent_tokenize(generations[i])

            if len(sents) == 0:
                ais_scores_all.append(0)
                ais_scores_prec_all.append(0)
                continue

            target_sents = [remove_citations(sent).strip() for sent in sents]

            entail = 0
            entail_prec = 0
            total_citations = 0
            for sent_id, sent in enumerate(sents):
                if len(target_sents[sent_id].strip()) == 0:
                    continue

                target_sent = target_sents[sent_id]  # Citation removed and (if opted for) decontextualized
                joint_entail = -1  # Undecided

                # Find references
                ref = [int(r[1:]) - 1 for r in re.findall(r"\[\d+", sent)]  # In text citation id starts from 1
                # logger.info(f"For `{sent}`, find citations {ref}")
                if len(ref) == 0:
                    # No citations
                    joint_entail = 0
                elif any([ref_id >= len(passages[i]) for ref_id in ref]):
                    # Citations out of range
                    joint_entail = 0
                else:
                    if at_most_citations is not None:
                        ref = ref[:at_most_citations]
                    total_citations += len(ref)
                    joint_passage = "\n".join([self._format_document(passages[i][psgs_id]) for psgs_id in ref])

                # If not directly rejected by citation format error, calculate the recall score
                if joint_entail == -1:
                    joint_entail = self._run_nli_autoais(joint_passage, target_sent)
                    autoais_log.append(
                        {
                            "question": questions[i],
                            "output": generations[i],
                            "claim": sent,
                            "passage": [joint_passage],
                            "model_type": "NLI",
                            "model_output": joint_entail,
                        }
                    )

                entail += joint_entail
                if len(ref) > 1:
                    sent_mcite += 1

                # calculate the precision score if applicable
                if joint_entail and len(ref) > 1:
                    sent_mcite_support += 1
                    # Precision check: did the model cite any unnecessary documents?
                    for psgs_id in ref:
                        # condition A
                        passage = self._format_document(passages[i][psgs_id])
                        nli_result = self._run_nli_autoais(passage, target_sent)

                        # condition B
                        if not nli_result:
                            subset_exclude = copy.deepcopy(ref)
                            subset_exclude.remove(psgs_id)
                            passage = "\n".join([self._format_document(passages[i][pid]) for pid in subset_exclude])
                            nli_result = self._run_nli_autoais(passage, target_sent)
                            if nli_result:  # psgs_id is not necessary
                                flag = 0
                                sent_mcite_overcite += 1
                            else:
                                entail_prec += 1
                        else:
                            entail_prec += 1

                    torch.cuda.empty_cache()
                else:
                    entail_prec += joint_entail

            sent_total += len(sents)
            ais_scores.append(entail / len(sents))
            ais_scores_prec.append(entail_prec / total_citations if total_citations > 0 else 0)  # len(sents))

            if sent_id in ais_scores_by_sentence:
                ais_scores_by_sentence[sent_id].append(entail / len(sents))
                ais_scores_prec_by_sentence[sent_id].append(
                    entail_prec / total_citations if total_citations > 0 else 0
                )
            else:
                ais_scores_by_sentence[sent_id] = [entail / len(sents)]
                ais_scores_prec_by_sentence[sent_id] = [entail_prec / total_citations if total_citations > 0 else 0]

            ais_scores_all.append(entail / len(sents))
            ais_scores_prec_all.append(entail_prec / total_citations if total_citations > 0 else 0)

        if sent_mcite > 0 and sent_mcite_support > 0:
            print(
                "Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite."
                % (
                    100 * sent_mcite / sent_total,
                    100 * sent_mcite_support / sent_mcite,
                    100 * sent_mcite_overcite / sent_mcite_support,
                )
            )

        return [
            {
                "citation_rec": 100 * np.mean(ais_scores),
                "citation_prec": 100 * np.mean(ais_scores_prec),
            },
            {"citation_rec_indiv": ais_scores_all, "citation_prec_indiv": ais_scores_prec_all},
            {
                "citation_rec_by_sentence": {
                    key: 100 * np.mean(value) for key, value in ais_scores_by_sentence.items()
                },
                "citation_prec_by_sentence": {
                    key: 100 * np.mean(value) for key, value in ais_scores_prec_by_sentence.items()
                },
            },
        ]

    def compute_shap_importance(self, passage, claim):
        # Only for single input, no batching possible at the moment

        encoded = self.autoais_tokenizer(passage, claim)["input_ids"][1:-1]
        decoded = self.autoais_tokenizer.decode(encoded)
        shap_values = self.explainer([decoded])

        scores_batch = shap_values[:, :, "consistent"].values
        data_batch_o = shap_values[:, :, "consistent"].data
        split_index = []
        data_batch = []
        score_lists = []

        # Slice data and retrieve index on which to split the passages from the retrieved statement.
        for i, entry in enumerate(data_batch_o):
            entry = entry.tolist()
            ind = (len(entry) - 1) - entry[::-1].index("</s>")
            split_index.append(ind)
            data_batch.append(data_batch_o[i][ind + 1 :])

        for i, entry in enumerate(scores_batch):
            scores = entry.tolist()[split_index[i] + 1 :]

            if self.shap_normalization == "min_max":
                if max(scores) == min(
                    scores
                ):  # In case the scores for all tokens is the same, avoid float zero divison
                    scores = scores
                else:
                    scores = [(x - min(scores)) / (max(scores) - min(scores)) for x in scores]

            score_list = [(x, data_batch[i][j]) for j, x in enumerate(scores)]

            score_lists.append(score_list)

        return score_lists[0]  # Only single element since no batching

    def compute_shap(self, generations, passages):

        generation_shaps = []

        for i in tqdm(range(len(generations))):
            # Get sentences by using NLTK
            shaps = []
            generations[i] = generations[i].replace(
                "</s>", " "
            )  # sometimes generates sentence end marker in output without ending?
            sents = sent_tokenize(generations[i])

            if len(sents) == 0:
                generation_shaps.append([])
                continue

            target_sents = [remove_citations(sent).strip() for sent in sents]

            for sent_id, sent in enumerate(sents):
                if len(target_sents[sent_id].strip()) == 0:
                    continue

                target_sent = target_sents[sent_id]  # Citation removed and (if opted for) decontextualized

                # Find references
                ref = [int(r[1:]) - 1 for r in re.findall(r"\[\d+", sent)]  # In text citation id starts from 1
                # logger.info(f"For `{sent}`, find citations {ref}")
                if len(ref) == 0 or any(
                    [ref_id >= len(passages[i]) for ref_id in ref]
                ):  # In case not attribution is given
                    joint_passage = "\n".join(
                        [self._format_document(passages[i][psgs_id]) for psgs_id in range(len(passages[i]))]
                    )
                    tokens_weight = self.compute_shap_importance(joint_passage, target_sent)
                    shaps.append(tokens_weight)
                else:
                    joint_passage = "\n".join([self._format_document(passages[i][psgs_id]) for psgs_id in ref])
                    tokens_weight = self.compute_shap_importance(joint_passage, target_sent)
                    shaps.append(tokens_weight)

            print(shaps)

            generation_shaps.append(shaps)

            # torch.cuda.empty_cache()

        return generation_shaps


def main():
    import os

    from src.data.template_formatter import TemplateFormatter

    # dataset = "asqa"
    dataset = "eli5"
    formatter = TemplateFormatter(dataset, setting_name="default", num_passages=3)
    consistency_model = AutoAIS(model_name="alignscore", threshold=0.9, rank=0, compute_shap=True)

    demos = formatter.demos
    samples = []

    generations = []
    passages = []
    for demo in demos:
        generation = demo["answer"]
        passages_sample = demo["docs"]
        generations.append(
            generation
        )  # Need to add the newline character so tokenization aligns with actual input to LLM
        passages.append(passages_sample)

    # generations = ["In 1961, Bobby Vee sang \"Take Good Care of My Baby\" [2]. And on the record were Barney Kessel, Tommy Allsup, and Howard Roberts on guitar, Clifford Hills on bass, Robert Florence on piano, and Earl Palmer on drums, while Sid Sharp did the string arrangements [3].</s>\u0434[\u0441]</s>\u0434[\u0441]"]
    # passages = [[{"id": "7966960", "title": "Take Good Care of My Baby", "text": "by Ralph Emery. It was released as a single on Liberty F-55383, in 1961. Another answer song, titled \"You Should Know I'm Still Your Baby\", was recorded by Sammi Lynn. It was released as a single on Sue Records 45-752, in 1961. Take Good Care of My Baby \"Take Good Care of My Baby\" is a song written by Carole King and Gerry Goffin. The song was made famous by Bobby Vee, when it was released in 1961. While searching for material for Bobby Vee to record, Vee's producer Snuff Garrett heard a demo of Carole King singing \"Take Good", "score": 0.8650405406951904}, {"id": "7966954", "title": "Take Good Care of My Baby", "text": "Take Good Care of My Baby \"Take Good Care of My Baby\" is a song written by Carole King and Gerry Goffin. The song was made famous by Bobby Vee, when it was released in 1961. While searching for material for Bobby Vee to record, Vee's producer Snuff Garrett heard a demo of Carole King singing \"Take Good Care of My Baby\". Garrett told publisher Don Kirshner that he wanted the song for Vee, but he believed the song needed an introductory verse. Garrett met with Carole King, and the introductory verse of Vee's version was written. Among the musicians", "score": 0.8504655361175537}, {"id": "7966955", "title": "Take Good Care of My Baby", "text": "on the record were Barney Kessel, Tommy Allsup, and Howard Roberts on guitar, Clifford Hills on bass, Robert Florence on piano, and Earl Palmer on drums, while Sid Sharp did the string arrangements. The Johnny Mann Singers sang backup. Bobby Vee released \"Take Good Care of My Baby\" as a single on July 20, 1961, and it was reviewed by \"Billboard\" in its issue dated July 31, 1961. Vee's recording quickly became popular, spending 15 weeks on the U.S. \"Billboard\" Hot 100, reaching No. 1 on September 21, 1961, and spending three weeks in that position. The song became a", "score": 0.8489639163017273}, {"id": "7966956", "title": "Take Good Care of My Baby", "text": "major hit internationally as well, reaching No. 1 in Canada, New Zealand, and the United Kingdom. The song was ranked No. 12 on \"Billboard\"s \"Hot 100 for 1961 - Top Sides of the Year\" and No. 23 on \"Cash Box\"s \"Top 100 Chart Hits of 1961\". The song was the lead track on Bobby Vee's album, \"Take Good Care of My Baby\", which was released in 1962. Vee re-recorded the song as a ballad in 1972. He released under his real name, Robert Thomas Velline, on his 1972 album \"Ain't Nothing Like a Sunny Day\", and as a single in", "score": 0.8427919745445251}, {"id": "7966957", "title": "Take Good Care of My Baby", "text": "1973. However, it is his original version, along with Bobby Vinton's, that remain as staples of oldies radio stations. In 1968, the song was released by Bobby Vinton as a single and on his album, \"Take Good Care of My Baby\". Vinton's version became a hit, spending 8 weeks on the U.S. \"Billboard\" Hot 100, reaching No. 33, while reaching No. 14 on \"Billboard\"s Easy Listening chart, No. 19 on \"Record World\"s \"100 Top Pops\", No. 12 on \"Record World\"s \"Top Non-Rock\" chart, No. 36 on Canada's \"\"RPM\" 100\", and No. 16 on Canada's CHUM Hit Parade. Vinton's version omitted", "score": 0.8389390110969543}]]

    # print(generations)
    # print(len(generations))

    shaps = consistency_model.compute_shap(generations, passages)

    print(shaps)

    shaps_dict = {}
    for i, ele in enumerate(shaps):
        shaps_dict["{}".format(i + 1)] = ele

    out_path = os.path.join("data", "{}".format(dataset), "shap_importance.json")
    with open(out_path, "w") as f_out:
        json.dump(shaps_dict, f_out, indent=4)

    print(shaps)

    # for demo in demos:
    # doc_list = []
    # for doc in demo["docs"][:3]:
    #     doc_prompt = formatter.doc_prompt
    #     text = doc['text']
    #     doc_prompt = doc_prompt.replace("{T}", doc["title"]).replace("{P}", text).replace("{ID}", str(doc_id+1))
    #     doc_list.append(doc_prompt)
    # passages = "".join(doc_list)
    # print(passages)


if __name__ == "__main__":
    main()


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--f", type=str, required=True, help="Output file. Should have field `question`, `output`, (ROUGE) `answer`, \
#                         (accuracy) `qa_pairs`, (AIS) `docs`")
#     parser.add_argument("--no_rouge", action="store_true", help="Do not evaluate ROUGE score")
#     parser.add_argument("--qa", action="store_true", help="Use the QA model")
#     parser.add_argument("--mauve", action="store_true", help="Use the mauve score model")
#     parser.add_argument("--citations", action="store_true", help="Evaluation with citation")
#     parser.add_argument("--at_most_citations", type=int, default=3, help="At most take this many documents (mostly for precision)")
#     parser.add_argument("--claims_nli", action="store_true", help="Use claims for ELI5")

#     # QAMPARI
#     parser.add_argument("--cot", action="store_true", help="For QAMPARI, try to find colon and separate the COT and answer listing")

#     args = parser.parse_args()

#     with open(args.f) as f:
#         data_with_config = json.load(f)
#     data = data_with_config['data']

#     if "qampari" in args.f:
#         args.no_rouge = True
#         args.qa = False
#         args.mauve = False
#         args.decontext = False
#         qampari = True
#     else:
#         qampari = False

#     # Truncate by newline and remove on the fly search result
#     logger.warning("We remove all the pre/appended space/newlines and we truncate the answer by the first newline.")
#     logger.warning("We replace any on the fly search result to standard bracket citation format.")
#     for i in range(len(data)):
#         data[i]['output'] = data[i]['output'].strip().split("\n")[0]
#         data[i]['output'] = data[i]['output'].replace("<|im_end|>", "")


#     # Remove all citations for all non-AutoAIS evaluation
#     normalized_data = copy.deepcopy(data)
#     for i in range(len(normalized_data)):
#         normalized_data[i]['output'] = remove_citations(normalized_data[i]['output'])

#     result = {}
#     result['length'] = compute_len(normalized_data)
#     result['str_em'], result['str_hit'] = compute_str_em(normalized_data)
#     if qampari:
#         result.update(compute_qampari_f1(normalized_data, cot=args.cot))
#     if not args.no_rouge:
#         result['rougeLsum'] = compute_rouge(normalized_data)
#     if args.qa:
#         result.update(compute_qa(normalized_data))
#     if args.mauve:
#         result['mauve'] = compute_mauve(normalized_data)
#     if args.citations:
#         result.update(compute_autoais(data, qampari=qampari, at_most_citations=args.at_most_citations))
#     if args.claims_nli:
#         result["claims_nli"] = compute_claims(normalized_data)

#     print(result)
#     json.dump(result, open(args.f + ".score", "w"), indent=4)
