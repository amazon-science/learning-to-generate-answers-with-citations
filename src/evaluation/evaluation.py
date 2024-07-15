import json
import logging
import os
import sys
import time
from functools import reduce

import torch
import transformers

from src.constants import ANSWER_KEY, GENERATION_KEY, ID_KEY, ORIGINAL_ID_KEY, QUESTION_KEY

# from src.evaluation.neural_metrics import
from src.evaluation.lexical_metrics import compute_qampari_f1, compute_rouge, compute_str_em, compute_str_em_normalized
from src.evaluation.neural_metrics import AutoAIS, compute_mauve, compute_qa, get_bert_score
from src.utils.util import remove_citations

# compute_qampari_f1,

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config, datamodule, main=False, rank=0):
        self.config = config
        self.datamodule = datamodule
        self.main = main
        self.rank = rank
        self.at_most_citations = 10
        if self.config.dataset == "asqa":
            # self.metrics = ["str_em", "rouge", "bert_score", "qa_metrics", "mauve", "AIS"]
            if self.config.normalized_scores:
                self.metrics = ["str_em", "rouge", "mauve", "AIS", "str_em_normalized"]
            else:
                self.metrics = ["str_em", "rouge", "mauve", "AIS"]
        elif self.config.dataset == "qampari":
            self.metrics = ["qampari_f1", "AIS"]
        elif self.config.dataset == "eli5":
            if self.config.normalized_scores:
                self.metrics = ["rouge", "mauve", "AIS", "eli5_claims", "eli5_claims_normalized"]
            else:
                self.metrics = ["rouge", "mauve", "AIS", "eli5_claims"]
        elif self.config.dataset == "hagrid":  # only used for training, not evaluation
            self.metrics = ["mauve", "AIS", "rouge"]
        elif self.config.dataset == "factscore":
            self.metrics = ["mauve", "AIS"]
        else:
            logger.error("Dataset {} not recognized by evaluator".format(self.config.dataset))
            sys.exit()

        # if (self.config.is_debug and not self.config.ws_attribution_training):
        #     self.metrics.remove("AIS")

    def filter_predictions_for_ws(self, scores, orig_ids):
        list_ids = []
        for i, entry in enumerate(scores):
            if (
                entry["citation_rec_indiv"] > self.config.ws_filtering_threshold
                and entry["citation_prec_indiv"] > self.config.ws_filtering_threshold
            ):
                if self.config.dataset == "asqa" and entry["str_em_indiv"] > (
                    self.config.ws_filtering_threshold_correctness
                ):
                    list_ids.append(orig_ids[i])
                elif (
                    self.config.dataset == "qampari"
                    and entry["qampari_f1_top5_indiv"] > (self.config.ws_filtering_threshold_correctness)
                    and entry["qampari_rec_top5_indiv"] > (self.config.ws_filtering_threshold_correctness)
                ):  # Maybe need to use not just top5 here?
                    list_ids.append(orig_ids[i])
                elif self.config.dataset == "eli5":  # No additional filtering requirement for now
                    list_ids.append(orig_ids[i])
                elif self.config.dataset == "hagrid":
                    list_ids.append(orig_ids[i])
        return list_ids

    def compute_metric(self, accumulated, split):
        logger.info("Computing metrics now...")

        results = {}
        filter_ids = []
        indiv_scores = {}

        if self.main:
            dataset = self.datamodule
        elif split == "train":
            dataset = self.datamodule.train_dataset.dataset_original
        elif split == "dev":
            dataset = self.datamodule.dev_dataset.dataset_original
        elif split == "test":
            dataset = self.datamodule.test_dataset.dataset_original  # For now
        else:
            logger.error("Split {} not known, aborting...".format(split))
            sys.exit()

        preds = accumulated[GENERATION_KEY]
        gold = accumulated[ANSWER_KEY]
        preds_no_citations = [remove_citations(x) for x in preds]

        tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.reader_model_origin)
        avg_gen_length = sum([len(tokenizer(pred)["input_ids"]) for pred in preds]) / len(preds)
        # avg_gold_length = sum([len(tokenizer(gol)["input_ids"]) for gol in gold]) / len(gold)

        results["gen_length"] = avg_gen_length
        # results["gold_length"] = avg_gold_length

        if "str_em" in self.metrics:
            qa_pairs = []
            for key in accumulated[ORIGINAL_ID_KEY]:
                qa_pairs.append(dataset.qa_pairs[key])
            score = compute_str_em(qa_pairs=qa_pairs, generations=preds_no_citations)
            results["str_em"], results["str_hit"], str_em_indiv, str_hit_indiv = score

        if "bert_score" in self.metrics:
            score = get_bert_score(gold, preds_no_citations)
            results["bert_score"] = str(score)

        if "qa_metrics" in self.metrics:
            qa_pairs = []
            for key in accumulated[ORIGINAL_ID_KEY]:
                qa_pairs.append(dataset.qa_pairs[key])

            score = compute_qa(qa_pairs=qa_pairs, generations=preds_no_citations)
            results.update(score)

        if "qampari_f1" in self.metrics:
            score, score_indiv_qampari = compute_qampari_f1(answers=gold, generations=preds_no_citations)
            results.update(score)

        if split == "dev" and "AIS" in self.metrics:  # Only run AtuoAIS evaluation on actual evaluation data
            torch.cuda.empty_cache()
            logger.info("Loading AutoAIS model, rank {}...".format(self.rank))
            # time.sleep(self.rank * 20)
            auto_ais = AutoAIS(rank=self.rank)  # REMOVE!

            passages = []
            for key in accumulated[ORIGINAL_ID_KEY]:
                passages.append(dataset.passages[key][: self.config.n_context])
            questions = accumulated[QUESTION_KEY]

            scores, indiv_scores, scores_by_sentence = auto_ais.compute_autoais(
                questions=questions,
                generations=preds,
                passages=passages,
                at_most_citations=self.at_most_citations,
                qampari=self.config.dataset == "qampari",
            )  # TODO Check that changing most citations from 5 to 10 does not result in perfromance changes on ASQA.
            results.update(scores)
            results["citation_rec_sentence"] = scores_by_sentence["citation_rec_by_sentence"]
            results["citation_prec_sentence"] = scores_by_sentence["citation_prec_by_sentence"]

            attributed_sentences = auto_ais.compute_attributed_sentences(
                questions=questions, generations=preds, passages=passages
            )

            if "eli5_claims" in self.metrics:
                claims = []
                for key in accumulated[ORIGINAL_ID_KEY]:
                    claims.append(dataset.claims[key])

                score = auto_ais.compute_claims(claims_all=claims, generations=preds)
                results["eli5_claims_nli"] = score

            # Normalized scores that only compute correctness over sentences that are attributed in retrieved passages.
            if "eli5_claims_normalized" in self.metrics:
                score = auto_ais.compute_claims_normalized(
                    claims_all=claims, generations=preds, attributed_sentences=attributed_sentences
                )
                results["eli5_claims_nli_normalized"] = score

            if "str_em_normalized" in self.metrics:
                for key in accumulated[ORIGINAL_ID_KEY]:
                    qa_pairs.append(dataset.qa_pairs[key])
                score = compute_str_em_normalized(
                    qa_pairs=qa_pairs, generations=preds_no_citations, attributed_sentences=attributed_sentences
                )
                results["str_em_normalized"], results["str_hit_normalized"], str_em_indiv_norm, str_hit_indiv_norm = (
                    score
                )

            del auto_ais
            auto_ais = None
            torch.cuda.empty_cache()

        if (
            split == "test"
            and self.config.ws_attribution_training
            and not ("t5_xxl" in self.config.consistency_model and indiv_scores)
        ):  # Only run consistency mdodel for filtering (i.e. "test" run)
            logger.info("Loading consistency model...")
            questions = accumulated[QUESTION_KEY]
            passages = []
            for key in accumulated[ORIGINAL_ID_KEY]:
                passages.append(dataset.passages[key][: self.config.n_context])
            consistency_model = AutoAIS(
                model_name=self.config.consistency_model, threshold=self.config.consistency_threshold, rank=self.rank
            )
            _, indiv_scores, _ = consistency_model.compute_autoais(
                questions=questions, generations=preds, passages=passages, at_most_citations=self.at_most_citations
            )

            del consistency_model
            consistency_model = None
            torch.cuda.empty_cache()

        scores_individual = []

        if split == "test" and self.config.ws_attribution_training:
            for i in range(len(accumulated[ORIGINAL_ID_KEY])):
                score_dict = {}
                score_dict["citation_rec_indiv"] = indiv_scores["citation_rec_indiv"][i]
                score_dict["citation_prec_indiv"] = indiv_scores["citation_prec_indiv"][i]
                if self.config.dataset == "asqa":
                    score_dict.update({"str_em_indiv": str_em_indiv[i], "str_hit_indiv": str_hit_indiv[i]})
                elif self.config.dataset == "qampari":
                    score_dict.update(
                        {
                            "qampari_prec_indiv": score_indiv_qampari["qampari_prec_indiv"][i],
                            "qampari_rec_indiv": score_indiv_qampari["qampari_rec_indiv"][i],
                            "qampari_rec_top5_indiv": score_indiv_qampari["qampari_rec_top5_indiv"][i],
                            "qampari_f1_indiv": score_indiv_qampari["qampari_f1_indiv"][i],
                            "qampari_f1_top5_indiv": score_indiv_qampari["qampari_f1_top5_indiv"][i],
                        }
                    )
                scores_individual.append(score_dict)

            filter_ids = self.filter_predictions_for_ws(scores_individual, accumulated[ID_KEY])

        logger.info(filter_ids)
        logger.info(results)

        return results, filter_ids

    def compute_and_save_shaps(self, accumulated, filtered_ids):
        # Only running shaps for test set, i.e. where we want to do augmentation
        dataset = self.datamodule.test_dataset.dataset_original  # For now

        logger.info("Loading consistency model...")
        preds = accumulated[GENERATION_KEY]
        passages_filtered = []
        generations_filtered = []
        ids_filtered = []
        for i, key in enumerate(accumulated[ID_KEY]):
            if key in filtered_ids:
                ids_filtered.append(key)
                generations_filtered.append(preds[i])
                passages_filtered.append(dataset.passages[accumulated[ORIGINAL_ID_KEY][i]][: self.config.n_context])

        consistency_model = AutoAIS(
            model_name=self.config.consistency_model,
            threshold=self.config.consistency_threshold,
            rank=self.rank,
            compute_shap=True,
            shap_normalization=self.config.shap_normalization,
        )
        shaps = consistency_model.compute_shap(generations=generations_filtered, passages=passages_filtered)

        shaps_dict = {}
        for i, ele in enumerate(ids_filtered):
            id_index = accumulated[ID_KEY].index(ele)

            shaps_dict[accumulated[ORIGINAL_ID_KEY][id_index]] = shaps[i]

        out_path = self.config.test_pred_file.replace(".json", "_filtered_shaps.json")
        with open(out_path, "w") as f_out:
            json.dump(shaps_dict, f_out, indent=4)

        del consistency_model
        consistency_model = None
        torch.cuda.empty_cache()

    def compute_rouge(self, accumulated, split):
        if self.main:
            dataset = self.datamodule
        elif split == "train":
            dataset = self.datamodule.train_dataset.dataset_original
        elif split == "dev":
            dataset = self.datamodule.dev_dataset.dataset_original
        elif split == "test":
            dataset = self.datamodule.test_dataset.dataset_original  # For now
        else:
            logger.error("Split {} not known, aborting...".format(split))
            sys.exit()

        preds = accumulated[GENERATION_KEY]
        gold = accumulated[ANSWER_KEY]

        preds_no_citations = [remove_citations(x) for x in preds]

        references1 = []
        references2 = None

        if self.config.dataset in ["asqa"]:
            references2 = []
            annotations = []
            for key in accumulated[ORIGINAL_ID_KEY]:
                annotations.append(dataset.annotations[key])

            for anno in annotations:
                references1.append(anno[0]["long_answer"])
                if len(anno) > 1:  # Not always more than one long answer
                    references2.append(anno[1]["long_answer"])
                else:
                    references2.append(anno[0]["long_answer"])
        else:
            references1 = gold

        if not references2:
            references2 = None

        score = compute_rouge(preds_no_citations, references1, references2)
        return score

    def compute_mauve(self, accumulated):
        questions = accumulated[QUESTION_KEY]
        preds = accumulated[GENERATION_KEY]
        gold = accumulated[ANSWER_KEY]

        preds_no_citations = [remove_citations(x) for x in preds]

        score = compute_mauve(questions=questions, answers=gold, generations=preds_no_citations, device_id=self.rank)

        torch.cuda.empty_cache()
        return score


def standalone_eval(metrics, accumulated, dataset_name, dataset, rank=0, at_most_citations=10):
    results = {}

    preds = accumulated[GENERATION_KEY]
    gold = accumulated[ANSWER_KEY]
    preds_no_citations = [remove_citations(x) for x in preds]

    if "str_em" in metrics:
        qa_pairs = []
        for key in accumulated[ORIGINAL_ID_KEY]:
            qa_pairs.append(dataset.qa_pairs[key])

        score = compute_str_em(qa_pairs=qa_pairs, generations=preds_no_citations)
        results["str_em"], results["str_hit"], str_em_indiv, str_hit_indiv = score

    if "rouge" in metrics:
        references1 = []
        references2 = None

        if dataset_name in ["asqa"]:
            references2 = []
            annotations = dataset.annotations

            for anno in annotations:
                references1.append(anno[0]["long_answer"])
                if len(anno) > 1:  # Not always more than one long answer
                    references2.append(anno[1]["long_answer"])
                else:
                    references2.append(anno[0]["long_answer"])
        else:
            references1 = gold

        if not references2:
            references2 = None

        score = compute_rouge(preds_no_citations, references1, references2)
        results["rouge"] = score

    if "bert_score" in metrics:
        score = get_bert_score(gold, preds_no_citations)
        results["bert_score"] = str(score)

    if "qa_metrics" in metrics:
        qa_pairs = dataset.qa_pairs

        score = compute_qa(qa_pairs=qa_pairs, generations=preds_no_citations)
        results.update(score)

    if "qampari_f1" in metrics:
        score, score_indiv_qampari = compute_qampari_f1(answers=gold, generations=preds_no_citations)
        results.update(score)

    if "AIS" in metrics:  # Only run AtuoAIS evaluation on actual evaluation data
        torch.cuda.empty_cache()
        logger.info("Loading AutoAIS model, rank {}...".format(rank))
        # time.sleep(self.rank * 20)
        auto_ais = AutoAIS(rank=rank)  # REMOVE!

        passages = dataset.passages
        questions = accumulated[QUESTION_KEY]
        scores, indiv_scores, scores_by_sentence = auto_ais.compute_autoais(
            questions=questions,
            generations=preds,
            passages=passages,
            at_most_citations=at_most_citations,
            qampari=dataset_name == "qampari",
        )  # TODO Check that changing most citations from 5 to 10 does not result in perfromance changes on ASQA.
        results.update(scores)

        results["citation_rec_sentence"] = scores_by_sentence["citation_rec_by_sentence"]
        results["citation_prec_sentence"] = scores_by_sentence["citation_prec_by_sentence"]

        attributed_sentences = auto_ais.compute_attributed_sentences(
            questions=questions, generations=preds, passages=passages
        )

        if "eli5_claims" in metrics:
            claims = []
            for key in accumulated[ORIGINAL_ID_KEY]:
                claims.append(dataset.claims[key])

            score = auto_ais.compute_claims(claims_all=claims, generations=preds)
            results["eli5_claims_nli"] = score

        # Normalized scores that only compute correctness over sentences that are attributed in retrieved passages.
        if "eli5_claims_normalized" in metrics:
            score = auto_ais.compute_claims_normalized(
                claims_all=claims, generations=preds, attributed_sentences=attributed_sentences
            )
            results["eli5_claims_nli_normalized"] = score

        if "str_em_normalized" in metrics:
            for key in accumulated[ORIGINAL_ID_KEY]:
                qa_pairs.append(dataset.qa_pairs[key])
            score = compute_str_em_normalized(
                qa_pairs=qa_pairs, generations=preds_no_citations, attributed_sentences=attributed_sentences
            )
            results["str_em_normalized"], results["str_hit_normalized"], str_em_indiv_norm, str_hit_indiv_norm = score

        del auto_ais
        auto_ais = None
        torch.cuda.empty_cache()
    return results


if "__main__" == __name__:
    import argparse

    from nltk import sent_tokenize

    class TempDataLoader:
        def __init__(self, path):
            self.questions = []
            self.answers = []
            self.generations = []
            self.original_ids = []
            self.passages = []
            self.qa_pairs = {}
            self.annotations = []

            with open(path, "r") as f_in:
                content = json.load(f_in)
                for sample in content:
                    # qid = sample["original_id"][0]
                    self.original_id = sample["original_id"]
                    self.questions.append(sample["question"])
                    self.answers.append(sample["answer"])
                    self.generations.append(sample["generation"])
                    self.original_ids.append(sample["sample_id"])
                    self.passages.append(sample["passages"][:3])
                    self.qa_pairs[sample["sample_id"]] = sample["qa_pairs"]
                    self.annotations.append(sample["annotations"])

            self.generations = [baseline_replacement(x) for x in self.generations]

    class TempDataLoader_SanityCheck:
        """
        Sanity-check result evaluation:
        Original repo (ALCE): {
            "str_em": 38.06434599156118,
            "str_hit": 12.341772151898734,
            "rougeLsum": 36.20937721915231,
            "mauve": 47.37043279022974,
            "citation_rec": 56.56746031746033,
            "citation_prec": 60.94409282700421
        }
        Ours: {
            'str_em': 38.06434599156118,
            'str_hit': 12.341772151898734,
            'rouge': 36.18787125488062,
            'mauve':  48.001931226499
            'citation_rec': 56.01893711070927,
            'citation_prec': 59.80133614627285,
        }
        Sanity check passed. Small difference due to seed variation (batch normalization) and quantized usage of T5-xxl.
        """

        def __init__(self, path, path2):
            self.questions = []
            self.answers = []
            self.generations = []
            self.original_ids = []
            self.passages = []
            self.qa_pairs = {}
            self.annotations = []
            self.qid = []
            self.claims = {}

            with open(path, "r") as f_in:
                content = json.load(f_in)
                for sample in content:
                    self.qid.append(sample["original_id"])
                    self.questions.append(sample["question"])
                    self.answers.append(sample["answer"])
                    # self.generations.append(sample["generation"])
                    self.original_ids.append(sample["sample_id"])
                    self.passages.append(sample["passages"][:5])
                    self.qa_pairs[sample["sample_id"]] = sample["qa_pairs"]
                    self.annotations.append(sample["annotations"])

            with open(path2, "r") as f_in:
                content = json.load(f_in)
                mapping = {}
                mapping_passages = {}
                mapping_claims = {}
                for sample in content["data"]:
                    # qid = sample["sample_id"]
                    qid = sample["question"]
                    mapping[qid] = sample["output"]
                    mapping_passages[qid] = sample["docs"]
                    if "claims" in sample:
                        mapping_claims[qid] = sample["claims"]

                # self.generations = [mapping[x] for x in self.qid]
                # self.passages = [mapping_passages[x] for x in self.qid]

                self.generations = [mapping[x] for x in self.questions]
                self.passages = [mapping_passages[x] for x in self.questions]
                if mapping_claims:
                    self.claims = {self.original_ids[i]: mapping_claims[x] for i, x in enumerate(self.questions)}

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset to use")
    parser.add_argument("--is_debug", action="store_true")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--input_path2", type=str)
    args = parser.parse_args()

    # config = Config(args.config_files, args.kwargs)

    # datamodule = TempDataLoader(args.input_path)
    if args.input_path2:
        datamodule = TempDataLoader_SanityCheck(args.input_path, args.input_path2)
    else:
        datamodule = TempDataLoader(args.input_path)

    accumulated = {
        QUESTION_KEY: datamodule.questions,
        ANSWER_KEY: datamodule.answers,
        ORIGINAL_ID_KEY: datamodule.original_ids,
        GENERATION_KEY: datamodule.generations,
    }

    if args.dataset == "asqa":
        metrics = ["str_em", "str_em_normalized", "rouge", "mauve", "AIS"]
    elif args.dataset == "eli5":
        metrics = ["rouge", "mauve", "AIS", "eli5_claims", "eli5_claims_normalized"]

    dataset_name = args.dataset
    rank = 0

    scores = standalone_eval(metrics, accumulated, dataset_name, datamodule, rank=rank)
    print(scores)

    datamodule.generations = [remove_citations(x) for x in datamodule.generations]

    score = compute_mauve(
        questions=datamodule.questions, answers=datamodule.answers, generations=datamodule.generations, device_id=0
    )
    print("Mauve: ", score)
