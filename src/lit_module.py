import copy
import json
import logging
import os
import sys
import traceback
from collections import defaultdict
from statistics import mean

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from deepspeed.utils import zero_to_fp32
from pytorch_lightning import LightningModule
from transformers import AutoModel, AutoTokenizer

from src.constants import (
    ANSWER_KEY,
    GENERATION_KEY,
    ID_KEY,
    LOGITS_KEY,
    ORIGINAL_ID_KEY,
    QUESTION_KEY,
)
from src.evaluation.evaluation import Evaluator
from src.generators import LLM, run_trivial_reader_baseline
from src.rallm import RALLM
from src.utils.get_optimizer import get_optimizer
from src.utils.get_scheduler import get_scheduler
from src.utils.util import postprocess_generation

logger = logging.getLogger(__name__)


class LitModule(LightningModule):
    def __init__(self, config, datamodule, split):
        super().__init__()
        self.config = config
        self.model = None

        self.datamodule = datamodule

        self.train_outputs = []
        self.validation_outputs = []
        self.test_outputs = []

        self.accumulated = {}

        self.use_deepspeed = self.config.compute_strategy.startswith("deepspeed")
        self.use_ddp = self.config.compute_strategy.startswith("ddp")

        if self.use_ddp:
            logger.info("Using DDP for distributed computation...")

        self._last_global_step_saved = -1

        if self.config.focused_learning:  # and split in ["train", "test"]:
            assert (
                self.config.mask_all_but_citations
            ), "To weight tokens you need to enable the flag mask_all_but_citations needs to be set to 1."
            if self.config.ws_iteration <= 1:
                in_path = os.path.join("data", self.config.dataset, "shap_importance.json")
                with open(in_path, "r") as f_in:
                    self.shap_importance = json.load(f_in)
            else:
                in_path = os.path.join("data", self.config.dataset, "shap_importance.json")
                with open(in_path, "r") as f_in:
                    self.shap_importance = json.load(f_in)

                exp_dir_last_iteration = self.config.exp_dir.replace(
                    "iter_{}".format(self.config.ws_iteration), "iter_{}".format(self.config.ws_iteration - 1)
                )
                in_path = os.path.join(exp_dir_last_iteration, "test_pred_filtered_shaps.json")
                with open(in_path, "r") as f_in:
                    shap_importance = json.load(f_in)
                    self.shap_importance.update(shap_importance)
            
            # self.shap_importance = {int(x): y for x,y in self.shap_importance.items()}

            self.filtered_samples_ratio_last_iteration = (
                len(self.shap_importance) - 4 # Init shap file always of size 4. 
            ) / self.config.unlabeled_samples_cutoff
        else:
            self.filtered_samples_ratio_last_iteration = 0

    def training_step(self, batch, batch_idx):

        if self.use_ddp:
            logger.error("Currently DDP training is not supported, only inference...")
            sys.exit()

        query = batch.get(QUESTION_KEY, [""])
        original_ids = batch.get(ORIGINAL_ID_KEY, [""])
        ids = [x.cpu().item() for x in batch.get(ID_KEY, [""])]

        answers = [self.datamodule.train_dataset.dataset_original.answers[orig_id] for orig_id in original_ids]
        passages = [self.datamodule.train_dataset.dataset_original.passages[orig_id] for orig_id in original_ids]

        query_enc = self.model.retriever_tokenize(query) if not self.config.use_file_passages else None

        if not self.config.use_file_passages:
            query_ids_retriever = query_enc["input_ids"].cuda()
            query_mask_retriever = query_enc["attention_mask"].cuda()
            retrieved_passages, _ = self.model.retrieve(
                self.index,
                self.config.n_context,
                query,
                query_ids_retriever,
                query_mask_retriever,
            )
        else:
            assert passages, "cant use use_file_passages mode without passing in passages"
            retrieved_passages = [p[: self.config.n_context] for p in passages]

        # TODO: Sometimes weird formatting breaks focused learning alignment. Fix more explicitly. VERY rare, but ignore for now. Add formatting sanity check before adding file to weakly attribtued data.
        try:
            reader_loss, retriever_loss = self.model(
                index=self.index,
                query_id=original_ids,
                query=query,
                target=answers,
                target_tokens=None,
                passages=retrieved_passages if self.config.use_file_passages else None,
                shap_values=(
                    self.shap_importance[original_ids[0]] if self.config.focused_learning else None
                ),  # TODO: CURRENTLY DOES NOT SUPPORT BATCHING
                batch_metadata=None,
                train_retriever=self.config.train_retriever,
            )
        except:
            print(traceback.print_exc())
            return None  # Does nothing if None is returned.

        if retriever_loss is not None and self.config.train_retriever:
            train_loss = reader_loss.float() + retriever_loss
        else:
            train_loss = reader_loss

        if not (self.use_deepspeed or self.use_ddp) or dist.get_rank() == 0:
            self.log("train_loss", train_loss.cpu().item(), prog_bar=True, on_step=True)

        if self.global_step % self.config.save_step_interval == 0:
            self.save_model()

        return train_loss

    def validation_step(self, batch, batch_idx):
        query = batch.get(QUESTION_KEY, [""])
        original_ids = batch.get(ORIGINAL_ID_KEY, [""])

        answers = [self.datamodule.dev_dataset.dataset_original.answers[orig_id] for orig_id in original_ids]
        passages = [self.datamodule.dev_dataset.dataset_original.passages[orig_id] for orig_id in original_ids]

        if not self.config.use_file_passages:
            query_enc = self.model.retriever_tokenize(query) if not self.config.use_file_passages else None
            query_ids_retriever = query_enc["input_ids"].cuda()
            query_mask_retriever = query_enc["attention_mask"].cuda()
            retrieved_passages, _ = self.model.retrieve(
                self.index,
                self.config.n_context,
                query,
                query_ids_retriever,
                query_mask_retriever,
            )
        else:
            assert passages, "cant use use_file_passages mode without passing in passages"
            retrieved_passages = [p[: self.config.n_context] for p in passages]

        if self.config.model_type == "decoder_only":
            reader_tokenizer = self.model.reader_tokenizer
            reader_tokens, _, _ = self.model.tokenize_passages_inference(query, answers, retrieved_passages)

            try:
                generation = self.model.generate(
                    reader_tokens,
                    query,
                    choices=None,
                    do_sample=self.config.generation_do_sample,
                )
                generations = []
                for k, g in enumerate(generation):
                    g = g.cpu()
                    logits = g[reader_tokens["input_ids"].size(1) :]
                    gen = reader_tokenizer.decode(logits)
                    gen = postprocess_generation(gen, self.config.reader_model_origin)
                    if self.config.answer_truncation in ["all", "inference"]:
                        gen = self.model.truncate_answer(gen, self.config.generation_max_length)
                    generations.append(gen)
                del generation
                # torch.cuda.empty_cache()
            except:
                generations = ["" for x in range(len(answers))]
                traceback.print_exc()

        elif self.config.model_type == "trivial_reader_baseline":
            generations = run_trivial_reader_baseline(
                retrieved_passages=retrieved_passages,
                max_length=self.config.generation_max_length,
                tokenizer=AutoTokenizer.from_pretrained(self.config.reader_model_origin),
            )

        output = {
            ID_KEY: [x.cpu().item() for x in batch.get(ID_KEY, [""])],
            ORIGINAL_ID_KEY: batch.get(ORIGINAL_ID_KEY, [""]),  # The id as used in the dataset
            QUESTION_KEY: batch.get(QUESTION_KEY, [""]),
            ANSWER_KEY: answers,
            GENERATION_KEY: generations,
        }

        self.validation_outputs.append(output)

    def test_step(self, batch, batch_idx):
        query = batch.get(QUESTION_KEY, [""])
        original_ids = batch.get(ORIGINAL_ID_KEY, [""])

        answers = [self.datamodule.test_dataset.dataset_original.answers[orig_id] for orig_id in original_ids]
        passages = [self.datamodule.test_dataset.dataset_original.passages[orig_id] for orig_id in original_ids]

        if not self.config.use_file_passages:
            query_enc = self.model.retriever_tokenize(query) if not self.config.use_file_passages else None
            query_ids_retriever = query_enc["input_ids"].cuda()
            query_mask_retriever = query_enc["attention_mask"].cuda()
            retrieved_passages, _ = self.model.retrieve(
                self.index,
                self.config.n_context,
                query,
                query_ids_retriever,
                query_mask_retriever,
            )
        else:
            assert passages, "cant use use_file_passages mode without passing in passages"
            retrieved_passages = [p[: self.config.n_context] for p in passages]

        if self.config.model_type == "decoder_only":
            reader_tokenizer = self.model.reader_tokenizer
            reader_tokens, _, above_max_length = self.model.tokenize_passages_inference(
                query, answers, retrieved_passages
            )

            if above_max_length:  # Only consider samples that are below max length otherwise, bad for weak supervision
                logger.info("Sample {} is too long. Skip...".format(query))
                return

            try:
                generation = self.model.generate(reader_tokens, query, choices=None, do_sample=True)

                generations = []
                for _, g in enumerate(generation):
                    g = g.cpu()
                    logits = g[reader_tokens["input_ids"].size(1) :]
                    gen = reader_tokenizer.decode(logits)
                    gen = postprocess_generation(gen, self.config.reader_model_origin)

                    if self.config.sample_citation_replacements:
                        generations += self.model.generate_alternative_candidates(gen, passages[0])
                    else:
                        generations.append(gen)

                del generation
            except:
                generations = ["" for x in range(len(answers))]
                traceback.print_exc()

        elif self.config.model_type == "trivial_reader_baseline":
            generations = run_trivial_reader_baseline(
                retrieved_passages=retrieved_passages,
                max_length=self.config.generation_max_length,
                tokenizer=AutoTokenizer.from_pretrained(self.config.reader_model_origin),
            )

        output = {
            ID_KEY: [x.cpu().item() for x in batch.get(ID_KEY, [""])],  # TODO: Only works for batch-size 1 right now
            ORIGINAL_ID_KEY: batch.get(ORIGINAL_ID_KEY, [""]),  # The id as used in the dataset
            QUESTION_KEY: batch.get(QUESTION_KEY, [""]),
            ANSWER_KEY: answers,
            GENERATION_KEY: generations,
        }

        if self.config.sample_citation_replacements:
            ids = [x.cpu().item() for x in batch.get(ID_KEY, [""])]
            # Duplicate entries by number of generation candidates
            output = {
                ID_KEY: [(self.config.unlabeled_samples_cutoff * ids[0]) + i + self.config.unlabeled_samples_cutoff * 10000 for i in range(1, len(generations) + 1)], # Creating unique ID representation. TODO: Only works for batch-size 1 right now.
                ORIGINAL_ID_KEY : batch.get(ORIGINAL_ID_KEY, [""])  * len(generations), # The id as used in the dataset
                QUESTION_KEY : batch.get(QUESTION_KEY, [""])  * len(generations),
                ANSWER_KEY : answers  * len(generations),
                GENERATION_KEY : generations,
            }

        self.test_outputs.append(output)

    def on_train_end(self):
        self.save_model(finish=True)

    def _accumulate_outputs(self, outputs):
        if self.use_deepspeed or self.use_ddp:
            gathered_outputs = [[] for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_outputs, outputs)
            outputs = [batch_output for outputs in gathered_outputs for batch_output in outputs]

        # let rank 0 collect all outputs
        accumulated = {key: [] for key in outputs[0].keys()}
        for batch_output in outputs:
            for key, value in batch_output.items():
                accumulated[key].extend(value)

        # multi-process may yield dupliated examples in the last batch
        valid_mask = []
        idx_set = set()
        for idx in accumulated[ID_KEY]:
            valid_mask.append(idx not in idx_set)
            idx_set.add(idx)
        for key, values in accumulated.items():
            accumulated[key] = [v for v, m in zip(values, valid_mask) if m]

        return accumulated

    def clean_model(self):
        del self.model
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        # exchange outputs between processes
        self.accumulated = self._accumulate_outputs(self.validation_outputs)

        self.clean_model()

        if not (self.use_deepspeed or self.use_ddp) or dist.get_rank() == 0:
            with open(self.config.dev_pred_file.replace(".json", "_{}.json".format(self.global_step)), "w") as f_out:
                validation_outputs_write = []
                for i in range(len(self.accumulated[QUESTION_KEY])):
                    write_dict = {}
                    sample_id = self.accumulated[ORIGINAL_ID_KEY][i]

                    qa_pairs = self.datamodule.dev_dataset.dataset_original.qa_pairs[sample_id]
                    passages = self.datamodule.dev_dataset.dataset_original.passages[
                        sample_id
                    ] 
                    annotations = self.datamodule.dev_dataset.dataset_original.annotations[sample_id]
                    write_dict["sample_id"] = sample_id
                    write_dict["passages"] = passages
                    write_dict["qa_pairs"] = qa_pairs
                    write_dict["annotations"] = annotations

                    write_dict[ORIGINAL_ID_KEY] = self.accumulated[ORIGINAL_ID_KEY][i]
                    write_dict[QUESTION_KEY] = self.accumulated[QUESTION_KEY][i]
                    write_dict[ANSWER_KEY] = self.accumulated[ANSWER_KEY][i]
                    write_dict[GENERATION_KEY] = self.accumulated[GENERATION_KEY][i]
                    validation_outputs_write.append(write_dict)
                json.dump(validation_outputs_write, f_out, indent=4)

        # if  not (self.use_deepspeed or self.use_ddp) or dist.get_rank() < self.config.evaluation_gpus:
        metrics, _ = self._distributed_evaluation(split="dev")

        if not (self.use_deepspeed or self.use_ddp) or dist.get_rank() == 0:
            result_str = json.dumps(metrics) + "\n"
            with open(self.config.dev_score_file, "a+") as f:
                f.write(result_str)
        else:
            metrics = {}

        self.validation_outputs = []
        self.accumulated = []

        torch.cuda.empty_cache()

        return metrics

    def _distributed_evaluation(self, split):
        """
        Accumulate results from GPUs.
        TODO: Mauve should not be computed by splitting dataset, consider evaluating seperately at once, but more diagnostic metric anyways.
        """
        if self.use_deepspeed or self.use_ddp:
            split_ids = [
                x.tolist() for x in np.array_split(self.accumulated[ORIGINAL_ID_KEY], torch.cuda.device_count())
            ]
            split_size = [len(x) for x in split_ids]

            accum_split = {}
            for key, value in self.accumulated.items():
                if key != ANSWER_KEY:  # answers not directly convertable to array->list
                    accum_split[key] = [x.tolist() for x in np.array_split(value, torch.cuda.device_count())][
                        dist.get_rank()
                    ]

            accum_split[ANSWER_KEY] = [
                x
                for i, x in enumerate(self.accumulated[ANSWER_KEY])
                if self.accumulated[ORIGINAL_ID_KEY][i] in split_ids[dist.get_rank()]
            ]

            metrics, filtered_ids = self.evaluator.compute_metric(accum_split, split=split)

            gathered_outputs = [[] for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_outputs, metrics)

            gathered_filtered_ids = [[] for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_filtered_ids, filtered_ids)

            if dist.get_rank() == 0:
                accumulated_metrics = {key: [] for key in list(gathered_outputs[0].keys())}
                for part in gathered_outputs:
                    for key, value in part.items():
                        accumulated_metrics[key].append(value)

                total_size = sum(split_size)
                for key, values in accumulated_metrics.items():
                    if isinstance(values[0], dict):
                        # metric is dict itself, i.e. sentence score
                        overall_score = {}
                        for i, part in enumerate(values):
                            for key2, value in part.items():
                                if key2 in overall_score:
                                    overall_score[key2].append(value * split_size[i])
                                else:
                                    overall_score[key2] = [value * split_size[i]]
                        overall_score = {key3: (sum(value) / total_size) for key3, value in overall_score.items()}
                        # overall_score = {key2: sum([x * split_size[i] for i,x in enumerate(value)]) / total_size for key2, value in values.items()}
                    else:
                        overall_score = sum([x * split_size[i] for i, x in enumerate(values)]) / total_size
                    accumulated_metrics[key] = overall_score

                if "citation_rec" in accumulated_metrics:
                    accumulated_metrics["citation_f1"] = (
                        2 * accumulated_metrics["citation_rec"] * accumulated_metrics["citation_prec"]
                    ) / (accumulated_metrics["citation_rec"] + accumulated_metrics["citation_prec"])
                # Need to compute mauve over all instances at once since metric does some clustering on ALL data
                if "mauve" in self.evaluator.metrics:
                    accumulated_metrics["mauve"] = self.evaluator.compute_mauve(self.accumulated)
                # Need to compute  rouge over all instances at once since cmputed at once, otherwise slight differences
                if "rouge" in self.evaluator.metrics:
                    accumulated_metrics["rouge"] = self.evaluator.compute_rouge(self.accumulated, split=split)

                accumulated_filtered_ids = []
                for filtered_ids in gathered_filtered_ids:
                    accumulated_filtered_ids += filtered_ids
                accumulated_filtered_ids = set(accumulated_filtered_ids)

                return [accumulated_metrics, accumulated_filtered_ids]
            else:
                return [[], []]
        else:
            metrics, filtered_ids = self.evaluator.compute_metric(self.accumulated, split=split)
            metrics["mauve"] = self.evaluator.compute_mauve(self.accumulated)
            metrics["rouge"] = self.evaluator.compute_rouge(self.accumulated, split=split)
            if "citation_rec" in metrics:
                metrics["citation_f1"] = (2 * metrics["citation_rec"] * metrics["citation_prec"]) / (
                    metrics["citation_rec"] + metrics["citation_prec"]
                )
            return [metrics, filtered_ids]

    def on_test_epoch_end(self):
        # exchange outputs between processes
        self.accumulated = self._accumulate_outputs(self.test_outputs)

        self.clean_model()

        if not (self.use_deepspeed or self.use_ddp) or dist.get_rank() == 0:
            with open(self.config.test_pred_file.replace(".json", "_{}.json".format(self.global_step)), "w") as f_out:
                test_outputs_write = []
                # print("Start writing...")
                # print("NUM SAMPLES {}".format(len(self.accumulated[QUESTION_KEY])))
                for i in range(len(self.accumulated[QUESTION_KEY])):
                    write_dict = {}
                    sample_id = self.accumulated[ORIGINAL_ID_KEY][i]
                    qa_pairs = self.datamodule.test_dataset.dataset_original.qa_pairs[sample_id]
                    passages = self.datamodule.test_dataset.dataset_original.passages[
                        sample_id
                    ]
                    annotations = self.datamodule.test_dataset.dataset_original.annotations[sample_id]

                    write_dict["sample_id"] = sample_id
                    write_dict["passages"] = passages
                    write_dict["qa_pairs"] = qa_pairs
                    write_dict["annotations"] = annotations

                    write_dict[ORIGINAL_ID_KEY] = self.accumulated[ORIGINAL_ID_KEY][i]
                    write_dict[QUESTION_KEY] = self.accumulated[QUESTION_KEY][i]
                    write_dict[ANSWER_KEY] = self.accumulated[ANSWER_KEY][i]
                    write_dict[GENERATION_KEY] = self.accumulated[GENERATION_KEY][i]
                    test_outputs_write.append(write_dict)
                json.dump(test_outputs_write, f_out, indent=4)

        # if not (self.use_deepspeed or self.use_ddp) or dist.get_rank() < self.config.evaluation_gpus:
        metrics, filtered_ids = self._distributed_evaluation(split="test")

        if not (self.use_deepspeed or self.use_ddp) or dist.get_rank() == 0:
            result_str = json.dumps(metrics) + "\n"
            with open(self.config.test_score_file, "a+") as f:
                f.write(result_str)

            if self.config.ws_attribution_training:
                with open(self.config.test_pred_file.replace(".json", "_filtered.jsonl"), "w") as f_out:
                    test_outputs_write = []
                    for i in range(len(self.accumulated[QUESTION_KEY])):
                        if self.accumulated[ID_KEY][i] in filtered_ids:

                            sample_id = self.accumulated[ORIGINAL_ID_KEY][i]
                            qa_pairs = self.datamodule.test_dataset.dataset_original.qa_pairs[sample_id]
                            passages = self.datamodule.test_dataset.dataset_original.passages[
                                sample_id
                            ]
                            annotations = self.datamodule.test_dataset.dataset_original.annotations[sample_id]

                            # Only consider filtered replacements if original is not also above threshold
                            if self.config.sample_citation_replacements and len([x for j, x in enumerate(self.accumulated[ORIGINAL_ID_KEY]) if self.accumulated[ORIGINAL_ID_KEY][j] == self.accumulated[ORIGINAL_ID_KEY][i] and self.accumulated[ID_KEY][j] in filtered_ids]) > 1 and min([self.accumulated[ID_KEY][j] for j, x in enumerate(self.accumulated[ID_KEY]) if self.accumulated[ORIGINAL_ID_KEY][j] == self.accumulated[ORIGINAL_ID_KEY][i] and self.accumulated[ID_KEY][j] in filtered_ids]) != self.accumulated[ID_KEY][i]:
                                logger.info("Skipping replacement instance since higher-scored instance of same original was found.") 
                                continue
                            write_dict = {}
                            write_dict["sample_id"] = sample_id
                            write_dict["answers"] = self.accumulated[GENERATION_KEY][i]
                            write_dict["passages"] = passages
                            write_dict["qa_pairs"] = qa_pairs
                            write_dict["annotations"] = annotations
                            write_dict["question"] = self.accumulated[QUESTION_KEY][i]

                            f_out.write("{}\n".format(json.dumps(write_dict)))

                            test_outputs_write.append(write_dict)

                with open(self.config.test_pred_file.replace(".json", "_filtered.json"), "w") as f_out_json:
                    json.dump(test_outputs_write, f_out_json, indent=4)

                if self.config.focused_learning:
                    self.evaluator.compute_and_save_shaps(self.accumulated, filtered_ids)

                filtered_samples_ratio = len(test_outputs_write) / self.config.unlabeled_samples_cutoff
                logger.info("Filtered ratio last iteration {}. This iteration the ratio is {}. and Stopping tolerence is {}".format(self.filtered_samples_ratio_last_iteration, filtered_samples_ratio, self.config.dynamic_stopping_tolerance))
                if filtered_samples_ratio + self.config.dynamic_stopping_tolerance < self.filtered_samples_ratio_last_iteration:
                    # Killing the entire bash script if condition has been met
                    import os
                    import signal
                    import sys

                    print(
                        "Stopping criterion for iterative training algorithm reached. Last iteration had a ratio of {} while this iteration has {}".format(
                            self.filtered_samples_ratio_last_iteration, filtered_samples_ratio
                        )
                    )
                    os.kill(os.getppid(), signal.SIGTERM)
                    sys.exit(1)

        else:
            metrics = {}

        self.test_outputs = []
        self.accumulated = {}

        torch.cuda.empty_cache()

        return metrics

    def configure_optimizers(self):
        optimizer, self.trainable_param_names = get_optimizer(self.model, self.config)
        scheduler = get_scheduler(optimizer, self.config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def init_retriever(self):
        if self.config.use_file_passages:
            return None, None

        contriever_encoder = AutoModel.from_pretrained("facebook/contriever")
        retriever_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

        # once you have done query side training you cannot go back to a parameter-tied retriever
        retriever_is_untied = self.config.query_side_retriever_training

        if retriever_is_untied:
            retriever = UntiedDualEncoderRetriever(self.config, contriever_encoder)
        else:
            retriever = DualEncoderRetriever(self.config, contriever_encoder)

        return retriever, retriever_tokenizer

    def init_reader(self):
        if self.config.model_type == "decoder_only":
            reader = LLM(config=self.config, rank=0 if not (self.use_deepspeed or self.use_ddp) else dist.get_rank())
            reader_tokenizer = AutoTokenizer.from_pretrained(self.config.reader_model_origin, use_fast=False)
            reader_tokenizer.padding_side = "left"
            return reader, reader.tokenizer
        else:
            logger.error("Cannot recognize model type {}".format(self.config.model_type))
            sys.exit()

    def _init_evaluator(self):
        if self.use_deepspeed or self.use_ddp:
            self.evaluator = Evaluator(self.config, self.datamodule, rank=dist.get_rank())
        else:
            self.evaluator = Evaluator(self.config, self.datamodule)

    def configure_model(self):
        if self.model is not None:
            return

        if self.config.model_type == "decoder_only":
            reader, reader_tokenizer = self.init_reader()
            retriever, retriever_tokenizer = self.init_retriever()
            self.model = RALLM(
                self.config, reader, retriever, reader_tokenizer, retriever_tokenizer
            )  # LoRA is intialized within model here, since using transformer package.
            self.load_model()
        else:
            logger.error(
                "Currently only decoder only model types are supported ,but found {}".format(self.config.model_type)
            )

        if self.config.use_file_passages:
            self.index = None
            self.passages = None
        else:
            logger.error(
                "Currently on-the-fly retrieval is not supported, so please set flag `use_file_passages` to true."
            )

        self._init_evaluator()

    def load_model(self):
        if self.config.load_weight != "":
            if self.config.model_type == "decoder_only":
                logger.info("Loading weights for model...")
                load_path = os.path.join(self.config.load_weight)
                trainable_states = torch.load(load_path, map_location=torch.device("cpu"))
                trainable_states = trainable_states
                load_result = self.model.load_state_dict(trainable_states, strict=False)
            else:
                logger.error(
                    "Currently only decoder only model types are supported ,but found {}".format(
                        self.config.model_type
                    )
                )

            logger.info(f"Model loaded from {self.config.load_weight}")
            logger.info(f"Unexpected keys {load_result.unexpected_keys.__str__()}")

    def save_model(self, finish=False):
        if self.config.save_model and finish:
            if finish:
                model_fname = os.path.join(self.config.exp_dir, "finish.pt")
            else:
                model_fname = os.path.join(self.config.exp_dir, f"global_step{self.global_step}.pt")

            if self.use_deepspeed or self.use_ddp:
                distributed_save_path = os.path.join(self.config.exp_dir, "saved_model")
                self.trainer.model.save_checkpoint(distributed_save_path)
                torch.distributed.barrier()
                if dist.get_rank() == 0:
                    trainable_states = zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(distributed_save_path)
                    prefix_length = len("module.model.")
                    trainable_states = {k[prefix_length:]: v for k, v in trainable_states.items()}
                    torch.save(trainable_states, model_fname)
            else:
                trainable_states = {
                    param_name: param_weight.cpu()
                    for param_name, param_weight in self.model.state_dict().items()
                    if param_name in self.trainable_param_names
                }
                torch.save(trainable_states, model_fname)

            self._last_global_step_saved = self.global_step
