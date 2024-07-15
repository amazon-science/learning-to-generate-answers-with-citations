import copy
import logging
import math
import random
import re
import sys
import time
from functools import reduce
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from nltk import sent_tokenize
from transformers import DataCollatorForLanguageModeling

from src.data.template_formatter import TemplateFormatter

logger = logging.getLogger(__name__)
BERT_MAX_SEQ_LENGTH: int = 512
CITATION_TOKENS = ["]", "],", "]:", "];", "].", "](", "])"]
SYSTEM_PROMPT = "You are a helpful and factually accurate assistant."


def encode_passages(batch, tokenizer, max_length):
    bsz = len(batch)
    n = max([len(example) for example in batch])
    batch = [example + [""] * (n - len(example)) for example in batch]
    batch = reduce(lambda a, b: a + b, batch)
    tokens = tokenizer(
        batch,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        truncation=True,
    )
    tokens = {k: v.view(bsz, n, -1) for k, v in tokens.items()}
    return tokens


class RALLM(nn.Module):
    def __init__(self, config, reader, retriever, reader_tokenizer, retriever_tokenizer):
        super(RALLM, self).__init__()

        self.reader = reader
        self.retriever = retriever
        self.reader_tokenizer = reader_tokenizer
        self.retriever_tokenizer = retriever_tokenizer
        self.config = config
        self.template_formatter = TemplateFormatter(
            dataset=config.dataset, setting_name=config.template_setting_name, num_passages=config.n_context
        )
        # )

        self.READER_ALL_TOKENS = list(self.reader_tokenizer.get_vocab().values())

        self.reader_tokenizer.pad_token = "[PAD]"
        # self.reader_tokenizer.pad_token = self.reader_tokenizer.unk_token

        self.reader_tokenizer.padding_side = "left"
        self.IGNORE_INDEX = -100

    def generate_alternative_candidates(self, generation, passages):

        passage_weights = [x["score"] for x in passages][:self.config.n_context]
        refs = [int(r[1:]) for r in re.findall(r"\[\d+", generation)]
        alt_responses = [generation]

        # Iterate through each citation
        for ref in refs:
            if random.randint(1, 3) <= 1:  # Randomly skip 30% of time
                continue
            elif len(alt_responses) >= 2:  # Sample only two alternative responses
                break
            alt_response = generation
            sampled_number = random.choices([1, 2, 3], weights=passage_weights)[0]  # Use retrieval score to sample alternative passage
            alt_response = alt_response.replace("[{}]".format(ref), f"[{sampled_number}]")
            if alt_response == generation:
                continue
            alt_responses.append(alt_response)
        return alt_responses

    def retriever_tokenize(self, query):
        if self.retriever_tokenizer:
            query_enc = self.retriever_tokenizer(
                query,
                max_length=min(self.config.text_maxlength, BERT_MAX_SEQ_LENGTH),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            query_enc = _to_cuda(query_enc)
        else:
            query_enc = None
        return _to_cuda(query_enc)

    def _map_tokenizer_weights(self, shap_weights):

        def continued_by(string_a, string_b, memory):
            for i in range(1, len(string_b)):
                if string_b[:-i] in string_a and "".join(memory) + string_b[:-i] == string_a:
                    return True
            return False

        exclude_tokens = ["<s>", "</s>", "<0x0A>"]

        id_to_weight_list = []
        sentences = [item for sublist in shap_weights for item in sublist]
        importance = [x[0] for x in sentences]
        toks = [x[1] for x in sentences]
        sentence = "\n" + "".join(toks)
        enc = self.reader.tokenizer(sentence, add_special_tokens=False)
        encodings = enc.tokens()

        shap_pointer = 0
        llm_pointer = 0

        llm_subwords = []
        shap_subwords = []
        shap_values = []
        memory_llm_subwords_pointer = []
        non_subset = False

        breaker = 0
        while shap_pointer < len(toks) and llm_pointer < len(encodings):
            shap_token = toks[shap_pointer].strip()
            llm_token = encodings[llm_pointer].replace("▁", " ").strip()

            if llm_token in exclude_tokens or llm_token == "":
                id_to_weight_list.append((0.0, encodings[llm_pointer]))
                llm_pointer += 1
                continue
            if shap_token in exclude_tokens or shap_token == "":
                shap_pointer += 1
                continue

            # If previously tokens were not direct subsets and stored aligned substrings, check if now completed
            if non_subset:
                if "".join(llm_subwords) == "".join(shap_subwords) + shap_token:
                    averaged_scores = sum(shap_values + [importance[shap_pointer]]) / (len(shap_values) + 1)
                    for entry in memory_llm_subwords_pointer:
                        id_to_weight_list.append((averaged_scores, encodings[entry]))
                    shap_subwords = []
                    shap_values = []
                    llm_subwords = []
                    memory_llm_subwords_pointer = []
                    shap_pointer += 1
                    non_subset = False
                    continue
                elif "".join(llm_subwords) + llm_token == "".join(shap_subwords):
                    averaged_scores = sum(shap_values) / len(shap_values)
                    for entry in memory_llm_subwords_pointer:
                        id_to_weight_list.append((averaged_scores, encodings[entry]))
                    id_to_weight_list.append((averaged_scores, encodings[llm_pointer]))
                    shap_subwords = []
                    shap_values = []
                    llm_subwords = []
                    memory_llm_subwords_pointer = []
                    llm_pointer += 1
                    non_subset = False
                    continue

            # If equal increment both
            if shap_token == llm_token:
                id_to_weight_list.append((importance[shap_pointer], encodings[llm_pointer]))
                llm_pointer += 1
                shap_pointer += 1

            # If shap pointer is a substring of llm pointer
            elif shap_token in llm_token:
                composed_word = "".join(shap_subwords) + shap_token
                if shap_subwords and composed_word == llm_token:
                    averaged_scores = sum(shap_values + [importance[shap_pointer]]) / (len(shap_values) + 1)
                    id_to_weight_list.append((averaged_scores, encodings[llm_pointer]))
                    shap_subwords = []
                    shap_values = []
                    shap_pointer += 1
                    llm_pointer += 1
                else:
                    shap_subwords.append(shap_token)
                    shap_values.append(importance[shap_pointer])
                    shap_pointer += 1

            # If llm pointer is a substring of shap pointer
            elif llm_token in shap_token:
                if "".join(llm_subwords) + llm_token == shap_token:
                    for entry in memory_llm_subwords_pointer:
                        id_to_weight_list.append((importance[shap_pointer], encodings[entry]))
                    id_to_weight_list.append((importance[shap_pointer], encodings[llm_pointer]))
                    shap_pointer += 1
                    llm_subwords = []
                    memory_llm_subwords_pointer = []
                else:
                    llm_subwords.append(llm_token)
                    memory_llm_subwords_pointer.append(llm_pointer)
                llm_pointer += 1

            # 'sports', 'person ' (shap), versus ▁sport', 'sp', 'erson (LLM)
            elif shap_token in "".join(llm_subwords):
                shap_subwords.append(shap_token)
                shap_values.append(importance[shap_pointer])
                shap_pointer += 1
                non_subset = True
            elif llm_token in "".join(shap_subwords):
                llm_subwords.append(llm_token)
                memory_llm_subwords_pointer.append(llm_pointer)
                llm_pointer += 1
                non_subset = True
            else:
                if continued_by(
                    "".join(shap_subwords) + shap_token, llm_token, llm_subwords
                ):  # eg. shap : 'Maw', 'syn', 'ram', ', and LLM is ▁M', 'aws', 'yn', 'ram', then we check that [M] + aws continues Maw potentially.
                    shap_subwords.append(shap_token)
                    shap_values.append(importance[shap_pointer])
                    shap_pointer += 1
                    non_subset = True
                elif continued_by("".join(llm_subwords) + llm_token, shap_token, "".join(shap_subwords)):
                    llm_subwords.append(llm_token)
                    memory_llm_subwords_pointer.append(llm_pointer)
                    llm_pointer += 1
                    non_subset = True
                else:
                    logger.error("PROBLEMATIC...WHAT IS GOING ON HERE...")
                    sys.exit()
        return id_to_weight_list

    def get_ids_of_citations(self, passage):
        citation_str = ["[{}]".format(i) for i in range(self.config.n_context)]
        citation_indices = []
        encodings = self.reader.tokenizer(passage)
        start_pos = None
        for i, token in enumerate(encodings.tokens()):
            if token.strip().replace("▁", "") == "[":
                start_pos = i
            if token.strip().replace("▁", "").replace(".", "") in CITATION_TOKENS and start_pos != None:
                for j in range((i - start_pos) + 1):
                    citation_indices.append(start_pos + j)
                start_pos = None

        non_citation_indices = []
        for i in range(len(encodings["input_ids"])):
            if i not in citation_indices:
                non_citation_indices.append(i)
        return non_citation_indices

    def weight_tokens(self, shap_values, reader_tok, labels_unmasked_or):
        import json
        import os

        labels_unmasked = [
            x for x in labels_unmasked_or[0].tolist() if x != -1
        ]  # TODO Currently batching is not supported for focused learning, fix!
        labels = reader_tok["labels"][0].tolist()[-len(labels_unmasked) :]

        assert len(labels_unmasked) == len(labels)

        labels_decoded = self.reader.tokenizer.decode(labels_unmasked)

        shap_importance_mapped = self._map_tokenizer_weights(shap_values)[
            2:
        ]  # First two indices are just some sentence beginning tokens, not neeed
        shap_importance_mapped_ids = [
            (x[0], self.reader.tokenizer.convert_tokens_to_ids(x[1])) for x in shap_importance_mapped
        ]  # 1592

        # Since labels have one additional token (eos_token) need to be added explcitly:
        shap_importance_mapped_ids.append((0.02, self.reader.tokenizer.eos_token_id))
        shap_importance_mapped_ids_w_citations = []

        citation_tokens_ids = [
            self.reader.tokenizer.encode(("\n" + x))[-1] for x in CITATION_TOKENS[1:]
        ]  # Get specific citation token
        citation_offset = 0

        # Incorporating citations might lead to erreneous token for spaces after citations, which should still be fine since we do only care about the importance per se, not the assigned token.
        # TODO: Sanity check
        for i, element in enumerate(labels_unmasked):  # exclude new line characters
            if labels[i] != self.IGNORE_INDEX:  # is citation
                citation_offset += 1
                shap_importance_mapped_ids_w_citations.append((self.config.importance_attribution_weights, labels[i]))
                # Adjust index if tokens are merged when having citations versus not having citations.
                # TODO: Quite hacky right now. Fix more generally.
                if labels[i] in citation_tokens_ids and not any(
                    [x in shap_importance_mapped[i - citation_offset][1] for x in [".", ",", "(", ")", ";", ":"]]
                ):  # labels[i] == 1592 or labels[i] == 9582: # Combined ] plus sentence end marker ".", so skip dot after that. Similar for ]: (9582).
                    citation_offset -= 1
            else:
                if (
                    self.config.focused_learning_mode == "hard_masking"
                    and shap_importance_mapped_ids[i - citation_offset][0] < self.config.importance_treshold
                ):
                    shap_importance_mapped_ids_w_citations.append(
                        (0.0, shap_importance_mapped_ids[i - citation_offset][1])
                    )
                else:
                    importance = shap_importance_mapped_ids[i - citation_offset]
                    if importance[0] == 0.0:  # Zero is bad for optimization...
                        importance = (0.1, importance[1])
                    shap_importance_mapped_ids_w_citations.append(importance)

        sanity_check_decoded = self.reader.tokenizer.decode([x[1] for x in shap_importance_mapped_ids_w_citations])

        labels_masked = []
        for element in shap_importance_mapped_ids_w_citations:
            if element[0] > self.config.importance_threshold:
                labels_masked.append(element[1])
            else:
                labels_masked.append(self.IGNORE_INDEX)

        prompt_length = len(reader_tok["labels"][0].tolist()) - len(labels)
        labels_masked = [-100] * prompt_length + labels_masked
        shap_importance_mapped_ids_w_citations = [
            (1.0, self.IGNORE_INDEX)
        ] * prompt_length + shap_importance_mapped_ids_w_citations

        assert len(labels_masked) == len(reader_tok["labels"][0].tolist())

        labels_masked = torch.tensor([labels_masked])

        if self.config.focused_learning_mode == "hard_masking":
            reader_tok["labels"] = labels_masked

        elif "soft_masking" in self.config.focused_learning_mode:  # Unmask citation, since we have the weights to use
            label_mask = labels_unmasked_or.ge(0)
            labels_unmasked_or[~label_mask] = self.IGNORE_INDEX
            # print(labels_unmasked_or)
            reader_tok["labels"] = labels_unmasked_or

        return reader_tok, [
            shap_importance_mapped_ids_w_citations
        ]  # List just to match mini-batching ,but always size 1, no proper batching supported yet.

    def truncate_answer(self, answer, max_length):
        sents = sent_tokenize(answer)
        curr_ans = []
        curr_len = 0
        for sent in sents:
            len_sent = len(self.reader.tokenizer(sent)["input_ids"])
            if (
                curr_len + len_sent > max_length and curr_len != 0
            ):  # ensure that at least one sentence is contained in answer
                break
            curr_len += len_sent
            curr_ans.append(sent)

        curr_ans = " ".join(curr_ans)
        return curr_ans

    def tokenize_passages_train(self, query_id, query, target, passages):
        if len(query) == 0:
            return None, None

        if self.config.answer_truncation in ["all", "train"]:
            target = [self.truncate_answer(x, self.config.train_max_length) for x in target]

        target = [x + self.reader.tokenizer.convert_ids_to_tokens(self.reader.tokenizer.eos_token_id) for x in target]

        if self.config.in_context_learning == "all":
            query_passages = self.template_formatter.apply_templates_to_samples_in_context(
                query,
                passages,
                num_training_samples=self.config.in_context_learning_samples,
                answers=target,
                attribution_representation=self.config.attribution_representation,
                query_id=query_id,
                use_chat_template=self.config.use_chat_template,
            )
        else:
            query_passages = self.template_formatter.apply_templates_to_samples(
                query,
                passages,
                answers=target,
                attribution_representation=self.config.attribution_representation,
                use_chat_template=self.config.use_chat_template,
            )

        if self.config.use_chat_template:
            if self.config.reader_model_origin in ["mistralai/Mistral-7B-Instruct-v0.1"]:
                chats = [[{"role": "user", "content": query_passage}] for query_passage in query_passages]
            else:
                chats = [
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query_passage}]
                    for query_passage in query_passages
                ]

            rendered_chats = [
                self.reader_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                for chat in chats
            ]
            # print(self.reader.tokenizer.eos_token_id)
            query_passages = [rendered_chat + target[i] for i, rendered_chat in enumerate(rendered_chats)]
            # <|im_end|>

        if self.config.mask_all_but_citations:
            ids_non_citations = [self.get_ids_of_citations(query_passage) for query_passage in query_passages]

        reader_tok_prompt = self.reader.tokenizer(
            [passage.replace(target[i], "") for i, passage in enumerate(query_passages)]
        )["input_ids"]
        reader_tok_example = self.reader.tokenizer(query_passages)["input_ids"]

        reader_tok_example = torch.tensor(reader_tok_example)

        labels = copy.deepcopy(reader_tok_example)
        labels_unmasked = copy.deepcopy(reader_tok_example)
        for i, label in enumerate(labels):
            labels[i][: len(reader_tok_prompt[i])] = -1
            labels_unmasked[i][: len(reader_tok_prompt[i])] = -1
            if self.config.mask_all_but_citations:  # Set all but the citation tokens to ignore
                labels[i][ids_non_citations[i]] = -1

        example_mask = reader_tok_example.ge(0)
        label_mask = labels.ge(0)
        reader_tok_example[~example_mask] = 0
        labels[~label_mask] = self.IGNORE_INDEX

        reader_tok = {
            "input_ids": reader_tok_example,  # .tolist(),
            "labels": labels,  # .tolist(),
            "attention_mask": example_mask,  # .tolist()
        }

        fstr = self.config.retriever_format
        retriever_passages = [[fstr.format(**p) for p in example] for example in passages]
        if self.retriever_tokenizer:
            retriever_tok = encode_passages(
                retriever_passages,
                self.retriever_tokenizer,
                min(self.config.text_maxlength, BERT_MAX_SEQ_LENGTH),
            )
            retriever_tok = _to_cuda(retriever_tok)
        else:
            retriever_tok = None

        decoder_input_ids = reader_tok_prompt

        reader_tok = _to_cuda(reader_tok)
        return reader_tok, labels_unmasked, retriever_tok

    def tokenize_passages_inference(self, query, target, passages):
        if len(query) == 0:
            return None, None

        if self.config.in_context_learning in ["all", "inference"] or (
            self.config.bootstrapping and self.config.ws_iteration <= 1
        ):
            query_passages = self.template_formatter.apply_templates_to_samples_in_context(
                query,
                passages,
                num_training_samples=self.config.in_context_learning_samples,
                answers=None,
                attribution_representation=self.config.attribution_representation,
            )
        else:
            query_passages = self.template_formatter.apply_templates_to_samples(
                query, passages, answers=None, attribution_representation=self.config.attribution_representation
            )

        if self.config.use_chat_template:
            if self.config.reader_model_origin in ["mistralai/Mistral-7B-Instruct-v0.1"]:
                chats = [[{"role": "user", "content": query_passage}] for query_passage in query_passages]
            else:
                chats = [
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query_passage}]
                    for query_passage in query_passages
                ]
            query_passages = [
                self.reader_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                for chat in chats
            ]

        reader_tok_check = self.reader_tokenizer(query_passages)
        prompt_length_check = [
            len(reader_tok_check["input_ids"][i]) for i in range(len(reader_tok_check["input_ids"]))
        ]
        length_above_max = any(
            [x > self.config.text_maxlength for x in prompt_length_check]
        )  # Just drop entire batch for now, batch size 2 normally so ok. TODO: Filter specific example that is too long

        logger.info(f"Prompt length={prompt_length_check}")

        if self.config.eval_batch_size > 1:
            assert (
                "llama" not in self.config.reader_model_origin.lower()
            ), "Currently batching with llama is not supported. Pad token breaks prediction."
            reader_tok = self.reader_tokenizer(
                query_passages,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.text_maxlength,
            )
        else:
            reader_tok = self.reader_tokenizer(
                query_passages, return_tensors="pt", truncation=True, max_length=self.config.text_maxlength
            )

        fstr = self.config.retriever_format
        retriever_passages = [[fstr.format(**p) for p in example] for example in passages]
        if self.retriever_tokenizer:
            retriever_tok = encode_passages(
                retriever_passages,
                self.retriever_tokenizer,
                min(self.config.text_maxlength, BERT_MAX_SEQ_LENGTH),
            )
            retriever_tok = _to_cuda(retriever_tok)
        else:
            retriever_tok = None

        reader_tok = _to_cuda(reader_tok)
        return reader_tok, retriever_tok, length_above_max

    def forward(
        self,
        index,
        query_id,
        query,
        target,
        target_tokens=None,
        passages=None,
        shap_values=None,
        batch_metadata=None,
        use_cache=False,
        train_retriever=False,
        iter_stats={},
    ):
        forward_start = time.time()
        bsz = len(query)

        query_enc = self.retriever_tokenize(query) if not self.config.use_file_passages else None

        if not self.config.use_file_passages:
            retrieve_start = time.time()
            passages, _ = self.retrieve(
                index,
                self.config.retriever_n_context,
                query,
                query_enc["input_ids"],
                query_enc["attention_mask"],
                batch_metadata=batch_metadata,
                iter_stats=iter_stats,
            )
            iter_stats["runtime/retrieve"] = (time.time() - retrieve_start, 1)

        reader_tokens, labels_unmasked, retriever_tokens = self.tokenize_passages_train(
            query_id, query, target, passages
        )

        if (
            self.config.focused_learning
        ):  # TODO: The flag whether to use focused learning or not should be an argument of the forward function
            reader_tokens, shap_importance_mapped_ids_w_citations = self.weight_tokens(
                shap_values, reader_tokens, labels_unmasked
            )

        reader_ids = reader_tokens["input_ids"]  # FIXME
        labels = reader_tokens["labels"]
        reader_mask = reader_tokens["attention_mask"].bool()

        n_context_training = min(self.config.n_context, reader_ids.size(1))

        retriever_loss = None

        if train_retriever:  # TODO: Currently not possible
            if self.config.use_gradient_checkpoint_retriever:
                self.retriever.gradient_checkpointing_enable()

            query_emb = self.retriever(**query_enc, is_passages=False)

            retriever_tokens = {k: v.reshape(-1, v.size(-1)) for k, v in retriever_tokens.items()}

            passage_emb = self.retriever(**retriever_tokens, is_passages=True).to(query_emb)
            passage_emb = passage_emb.view(bsz, -1, passage_emb.size(-1))
            retriever_score = torch.einsum("id, ijd->ij", [query_emb, passage_emb])

            if self.config.use_gradient_checkpoint_retriever:
                self.retriever.gradient_checkpointing_disable()

            # TODO: How to get decoder input ids here?
            if "ppmean" in self.config.gold_score_mode:
                gold_score = self.perplexity_score(reader_ids, reader_mask, decoder_input_ids, labels, bsz)

            if self.training:
                self.reader.train()

        reader_ids_training = reader_ids[:, :n_context_training].contiguous()
        reader_mask_training = reader_mask[:, :n_context_training].contiguous()

        reader_ids_training = reader_ids_training.view(reader_ids.size(0), -1)
        reader_mask_training = reader_mask_training.view(reader_mask.size(0), -1)

        reader_output = self.reader(
            **reader_tokens,
            importance_scores=shap_importance_mapped_ids_w_citations if self.config.focused_learning else None,
            use_cache=False,
        )
        reader_loss = reader_output[0]

        if train_retriever:
            retriever_score = retriever_score / np.sqrt(query_emb.size(-1))

            if gold_score is not None:
                gold_score = gold_score.float()
                retriever_score = retriever_score.float()
                if self.config.gold_score_mode == "emdr":
                    retriever_loss = self.logprob(retriever_score, gold_score, labels)
                else:
                    retriever_loss = self.kldivloss(retriever_score, gold_score)

        iter_stats["loss/reader_loss"] = (reader_loss.item(), len(query))
        if retriever_loss is not None:
            iter_stats["loss/retriever_loss"] = (retriever_loss.item(), len(query))

        iter_stats["runtime/forward"] = (time.time() - forward_start, 1)
        return reader_loss, retriever_loss

    @torch.no_grad()
    def generate(self, tokens, query, choices=None, do_sample=True):
        if (
            "llama" in self.config.reader_model_origin.lower()
            or "deci" in self.config.reader_model_origin.lower()
            or "mistral" in self.config.reader_model_origin.lower()
        ):
            stop = []  # if stop is None else stop
            stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"]))  # In Llama \n is <0x0A>; In OPT \n is Ċ
            # stop = list(set(stop + ["\n", "\n\n\n", "\n\n"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
            stop_token_ids = list(
                set(
                    [self.reader_tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop]
                    + [self.reader.model.config.eos_token_id]
                    + [self.reader.tokenizer.eos_token_id]
                )
            )
            stop_token_ids.remove(self.reader_tokenizer.unk_token_id)

        if (
            "llama" in self.config.reader_model_origin.lower()
            or "mistral" in self.config.reader_model_origin.lower()
            or "deci" in self.config.reader_model_origin.lower()
        ):
            max_length = 4096
            max_new_tokens = min(self.config.generation_max_length, max_length - len(tokens))
        else:
            max_length = 2048
            max_new_tokens = min(self.config.generation_max_length, max_length - len(tokens))

        if do_sample:
            outputs = self.reader.generate(
                **tokens,
                do_sample=do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_new_tokens=max_new_tokens,  # num_beams=4, num_beam_groups=2, diversity_penalty=1.0,
                num_return_sequences=1,
                num_beams=self.config.generation_num_beams,
                length_penalty=self.config.generation_length_penalty,
                eos_token_id=(
                    stop_token_ids if self.config.use_chat_template else None
                ),  # if "llama" in self.config.reader_model_origin.lower() else None
                pad_token_id=self.reader.tokenizer.eos_token_id,  # TODO: JUST ADDED, CHECK IT DOES NOT BREAK CODE.
            )
        else:
            outputs = self.reader.generate(
                **tokens,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                num_beams=self.config.generation_num_beams,
                length_penalty=self.config.generation_length_penalty,
                eos_token_id=(
                    stop_token_ids if self.config.use_chat_template else None
                ),  # if "llama" in self.config.reader_model_origin.lower() else None
                pad_token_id=self.reader.tokenizer.eos_token_id,  # TODO: JUST ADDED, CHECK IT DOES NOT BREAK CODE.
            )

        return outputs


def _to_cuda(tok_dict):
    return {k: v.cuda() for k, v in tok_dict.items()}
