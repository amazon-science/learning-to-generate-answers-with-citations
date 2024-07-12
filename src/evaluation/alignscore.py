"""
Copyrights: https://github.com/yuh-zha/AlignScore/
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import spacy
import torch
import torch.nn as nn
from nltk.tokenize import sent_tokenize
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import (
    AdamW,
    AlbertForMaskedLM,
    AlbertModel,
    AutoConfig,
    AutoTokenizer,
    BertForPreTraining,
    BertModel,
    RobertaForMaskedLM,
    RobertaModel,
    get_linear_schedule_with_warmup,
)


logger = logging.getLogger(__name__)


class AlignScore:
    def __init__(
        self, model: str, batch_size: int, device: int, ckpt_path: str, evaluation_mode="nli_sp", verbose=True
    ) -> None:
        self.model = Inferencer(
            ckpt_path=ckpt_path, model=model, batch_size=batch_size, device=device, verbose=verbose
        )
        self.model.nlg_eval_mode = evaluation_mode

    def score(self, contexts: List[str], claims: List[str]) -> List[float]:
        return self.model.nlg_eval(contexts, claims)[1].tolist()


class Inferencer:
    def __init__(self, ckpt_path, model="bert-base-uncased", batch_size=32, device="cuda", verbose=True) -> None:
        self.device = device
        if ckpt_path is not None:
            self.model = BERTAlignModel.load_from_checkpoint(checkpoint_path=ckpt_path, strict=False).to(self.device)
        else:
            logger.warning("Loading UNTRAINED alignment model!")
            self.model = BERTAlignModel(model=model).to(self.device)
        self.model.eval()
        self.batch_size = batch_size

        self.config = AutoConfig.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.spacy = spacy.load("en_core_web_sm")

        self.loss_fct = nn.CrossEntropyLoss(reduction="none")
        self.softmax = nn.Softmax(dim=-1)

        self.smart_type = "smart-n"
        self.smart_n_metric = "f1"

        self.disable_progress_bar_in_inference = False

        self.nlg_eval_mode = None  # bin, bin_sp, nli, nli_sp
        self.verbose = verbose

    def inference_example_batch(self, premise: list, hypo: list):
        """
        inference a example,
        premise: list
        hypo: list
        using self.inference to batch the process

        SummaC Style aggregation
        """
        self.disable_progress_bar_in_inference = True
        assert len(premise) == len(hypo), "Premise must has the same length with Hypothesis!"

        out_score = []
        for one_pre, one_hypo in tqdm(
            zip(premise, hypo), desc="Evaluating", total=len(premise), disable=(not self.verbose)
        ):
            out_score.append(self.inference_per_example(one_pre, one_hypo))

        return None, torch.tensor(out_score), None

    def inference_per_example(self, premise: str, hypo: str):
        """
        inference a example,
        premise: string
        hypo: string
        using self.inference to batch the process
        """

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield " ".join(lst[i : i + n])

        premise_sents = sent_tokenize(premise)
        premise_sents = premise_sents or [""]

        n_chunk = len(premise.strip().split()) // 350 + 1
        n_chunk = max(len(premise_sents) // n_chunk, 1)
        premise_sents = [each for each in chunks(premise_sents, n_chunk)]

        hypo_sents = sent_tokenize(hypo)

        premise_sent_mat = []
        hypo_sents_mat = []
        for i in range(len(premise_sents)):
            for j in range(len(hypo_sents)):
                premise_sent_mat.append(premise_sents[i])
                hypo_sents_mat.append(hypo_sents[j])

        if self.nlg_eval_mode is not None:
            if self.nlg_eval_mode == "nli_sp":
                output_score = self.inference(premise_sent_mat, hypo_sents_mat)[2][
                    :, 0
                ]  ### use NLI head OR ALIGN head
            elif self.nlg_eval_mode == "bin_sp":
                output_score = self.inference(premise_sent_mat, hypo_sents_mat)[1]  ### use NLI head OR ALIGN head
            elif self.nlg_eval_mode == "reg_sp":
                output_score = self.inference(premise_sent_mat, hypo_sents_mat)[0]  ### use NLI head OR ALIGN head

            output_score = (
                output_score.view(len(premise_sents), len(hypo_sents)).max(dim=0).values.mean().item()
            )  ### sum or mean depends on the task/aspect
            return output_score

        output_score = self.inference(premise_sent_mat, hypo_sents_mat)[2][:, 0]  ### use NLI head OR ALIGN head
        output_score = (
            output_score.view(len(premise_sents), len(hypo_sents)).max(dim=0).values.mean().item()
        )  ### sum or mean depends on the task/aspect

        return output_score

    def inference(self, premise, hypo):
        """
        inference a list of premise and hypo

        Standard aggregation
        """
        if isinstance(premise, str) and isinstance(hypo, str):
            premise = [premise]
            hypo = [hypo]

        batch = self.batch_tokenize(premise, hypo)
        output_score_reg = []
        output_score_bin = []
        output_score_tri = []
        output_attentions = []

        for mini_batch in tqdm(
            batch, desc="Evaluating", disable=not self.verbose or self.disable_progress_bar_in_inference
        ):
            mini_batch = mini_batch.to(self.device)
            with torch.no_grad():

                # saliency = self.compute_saliency_maps(premise, hypo)

                model_output = self.model(mini_batch)
                model_output_reg = model_output.reg_label_logits.cpu()
                model_output_bin = model_output.seq_relationship_logits  # Temperature Scaling / 2.5
                model_output_tri = model_output.tri_label_logits
                model_output_attentions = model_output.attentions[-1]  # attention of the last layer only

                model_output_bin = self.softmax(model_output_bin).cpu()
                model_output_tri = self.softmax(model_output_tri).cpu()
            output_score_reg.append(model_output_reg[:, 0])
            output_score_bin.append(model_output_bin[:, 1])
            output_score_tri.append(model_output_tri[:, :])
            output_attentions.append(model_output_attentions)

        output_score_reg = torch.cat(output_score_reg)
        output_score_bin = torch.cat(output_score_bin)
        output_score_tri = torch.cat(output_score_tri)

        if self.nlg_eval_mode is not None:
            if self.nlg_eval_mode == "nli":
                output_score_nli = output_score_tri[:, 0]
                return None, output_score_nli, None
            elif self.nlg_eval_mode == "bin":
                return None, output_score_bin, None
            elif self.nlg_eval_mode == "reg":
                return None, output_score_reg, None
            else:
                ValueError("unrecognized nlg eval mode")

        return output_score_reg, output_score_bin, output_score_tri, output_attentions

    def inference_reg(self, premise, hypo):
        """
        inference a list of premise and hypo

        Standard aggregation
        """
        self.model.is_reg_finetune = True
        if isinstance(premise, str) and isinstance(hypo, str):
            premise = [premise]
            hypo = [hypo]

        batch = self.batch_tokenize(premise, hypo)
        output_score = []

        for mini_batch in tqdm(batch, desc="Evaluating", disable=self.disable_progress_bar_in_inference):
            mini_batch = mini_batch.to(self.device)
            with torch.no_grad():
                model_output = self.model(mini_batch).seq_relationship_logits.cpu().view(-1)
            output_score.append(model_output)
        output_score = torch.cat(output_score)
        return output_score

    def batch_tokenize(self, premise, hypo):
        """
        input premise and hypos are lists
        """
        assert isinstance(premise, list) and isinstance(hypo, list)
        assert len(premise) == len(hypo), "premise and hypo should be in the same length."

        batch = []
        for mini_batch_pre, mini_batch_hypo in zip(
            self.chunks(premise, self.batch_size), self.chunks(hypo, self.batch_size)
        ):
            try:
                mini_batch = self.tokenizer(
                    mini_batch_pre,
                    mini_batch_hypo,
                    truncation="only_first",
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                )
            except:
                logger.warning("text_b too long...")
                mini_batch = self.tokenizer(
                    mini_batch_pre,
                    mini_batch_hypo,
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                )
            batch.append(mini_batch)

        return batch

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def nlg_eval(self, premise, hypo):
        assert self.nlg_eval_mode is not None, "Select NLG Eval mode!"
        if (self.nlg_eval_mode == "bin") or (self.nlg_eval_mode == "nli") or (self.nlg_eval_mode == "reg"):
            return self.inference(premise, hypo)

        elif (self.nlg_eval_mode == "bin_sp") or (self.nlg_eval_mode == "nli_sp") or (self.nlg_eval_mode == "reg_sp"):
            return self.inference_example_batch(premise, hypo)

        else:
            ValueError("Unrecognized NLG Eval mode!")


class BERTAlignModel(pl.LightningModule):
    def __init__(self, model="bert-base-uncased", using_pretrained=True, *args, **kwargs) -> None:
        super().__init__()
        # Already defined in lightning: self.device
        self.save_hyperparameters()
        self.model = model

        if "muppet" in model:
            assert using_pretrained == True, "Only support pretrained muppet!"
            self.base_model = RobertaModel.from_pretrained(model)
            self.mlm_head = RobertaForMaskedLM(AutoConfig.from_pretrained(model)).lm_head

        elif "roberta" in model:
            if using_pretrained:
                self.base_model = RobertaModel.from_pretrained(model, output_attentions=True)
                self.mlm_head = RobertaForMaskedLM.from_pretrained(model).lm_head
            else:
                self.base_model = RobertaModel(AutoConfig.from_pretrained(model))
                self.mlm_head = RobertaForMaskedLM(AutoConfig.from_pretrained(model)).lm_head

        elif "albert" in model:
            if using_pretrained:
                self.base_model = AlbertModel.from_pretrained(model)
                self.mlm_head = AlbertForMaskedLM.from_pretrained(model).predictions
            else:
                self.base_model = AlbertModel(AutoConfig.from_pretrained(model))
                self.mlm_head = AlbertForMaskedLM(AutoConfig.from_pretrained(model)).predictions

        elif "bert" in model:
            if using_pretrained:
                self.base_model = BertModel.from_pretrained(model)
                self.mlm_head = BertForPreTraining.from_pretrained(model).cls.predictions
            else:
                self.base_model = BertModel(AutoConfig.from_pretrained(model))
                self.mlm_head = BertForPreTraining(AutoConfig.from_pretrained(model)).cls.predictions

        elif "electra" in model:
            self.generator = BertModel(AutoConfig.from_pretrained("prajjwal1/bert-small"))
            self.generator_mlm = BertForPreTraining(AutoConfig.from_pretrained("prajjwal1/bert-small")).cls.predictions

            self.base_model = BertModel(AutoConfig.from_pretrained("bert-base-uncased"))
            self.discriminator_predictor = ElectraDiscriminatorPredictions(self.base_model.config)

        self.bin_layer = nn.Linear(self.base_model.config.hidden_size, 2)
        self.tri_layer = nn.Linear(self.base_model.config.hidden_size, 3)
        self.reg_layer = nn.Linear(self.base_model.config.hidden_size, 1)

        self.dropout = nn.Dropout(p=0.1)

        self.need_mlm = True
        self.is_finetune = False
        self.mlm_loss_factor = 0.5

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, batch):
        if "electra" in self.model:
            return self.electra_forward(batch)
        base_model_output = self.base_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"] if "token_type_ids" in batch.keys() else None,
        )

        prediction_scores = self.mlm_head(base_model_output.last_hidden_state)  ## sequence_output for mlm
        seq_relationship_score = self.bin_layer(
            self.dropout(base_model_output.pooler_output)
        )  ## pooled output for classification
        tri_label_score = self.tri_layer(self.dropout(base_model_output.pooler_output))
        reg_label_score = self.reg_layer(base_model_output.pooler_output)

        total_loss = None
        if "mlm_label" in batch.keys():  ### 'mlm_label' and 'align_label' when training
            ce_loss_fct = nn.CrossEntropyLoss(reduction="sum")
            masked_lm_loss = ce_loss_fct(
                prediction_scores.view(-1, self.base_model.config.vocab_size), batch["mlm_label"].view(-1)
            )  # / self.con vocabulary
            next_sentence_loss = ce_loss_fct(
                seq_relationship_score.view(-1, 2), batch["align_label"].view(-1)
            ) / math.log(2)
            tri_label_loss = ce_loss_fct(tri_label_score.view(-1, 3), batch["tri_label"].view(-1)) / math.log(3)
            reg_label_loss = self.mse_loss(reg_label_score.view(-1), batch["reg_label"].view(-1), reduction="sum")

            masked_lm_loss_num = torch.sum(batch["mlm_label"].view(-1) != -100)
            next_sentence_loss_num = torch.sum(batch["align_label"].view(-1) != -100)
            tri_label_loss_num = torch.sum(batch["tri_label"].view(-1) != -100)
            reg_label_loss_num = torch.sum(batch["reg_label"].view(-1) != -100.0)

        return ModelOutput(
            loss=total_loss,
            all_loss=(
                [masked_lm_loss, next_sentence_loss, tri_label_loss, reg_label_loss]
                if "mlm_label" in batch.keys()
                else None
            ),
            loss_nums=(
                [masked_lm_loss_num, next_sentence_loss_num, tri_label_loss_num, reg_label_loss_num]
                if "mlm_label" in batch.keys()
                else None
            ),
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            tri_label_logits=tri_label_score,
            reg_label_logits=reg_label_score,
            hidden_states=base_model_output.hidden_states,
            attentions=base_model_output.attentions,
        )

    def electra_forward(self, batch):
        if "mlm_label" in batch.keys():
            ce_loss_fct = nn.CrossEntropyLoss()
            generator_output = self.generator_mlm(
                self.generator(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"] if "token_type_ids" in batch.keys() else None,
                ).last_hidden_state
            )
            masked_lm_loss = ce_loss_fct(
                generator_output.view(-1, self.generator.config.vocab_size), batch["mlm_label"].view(-1)
            )

            hallucinated_tokens = batch["input_ids"].clone()

            hallucinated_tokens[batch["mlm_label"] != -100] = torch.argmax(generator_output, dim=-1)[
                batch["mlm_label"] != -100
            ]
            replaced_token_label = (
                batch["input_ids"] == hallucinated_tokens
            ).long()  # .type(torch.LongTensor) #[batch['mlm_label'] == -100] = -100
            replaced_token_label[batch["mlm_label"] != -100] = (batch["mlm_label"] == hallucinated_tokens)[
                batch["mlm_label"] != -100
            ].long()
            replaced_token_label[batch["input_ids"] == 0] = -100  ### ignore paddings

        base_model_output = self.base_model(
            input_ids=hallucinated_tokens if "mlm_label" in batch.keys() else batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"] if "token_type_ids" in batch.keys() else None,
        )
        hallu_detect_score = self.discriminator_predictor(base_model_output.last_hidden_state)
        seq_relationship_score = self.bin_layer(
            self.dropout(base_model_output.pooler_output)
        )  ## pooled output for classification
        tri_label_score = self.tri_layer(self.dropout(base_model_output.pooler_output))
        reg_label_score = self.reg_layer(base_model_output.pooler_output)

        total_loss = None

        if "mlm_label" in batch.keys():  ### 'mlm_label' and 'align_label' when training
            total_loss = []
            ce_loss_fct = nn.CrossEntropyLoss()
            hallu_detect_loss = ce_loss_fct(hallu_detect_score.view(-1, 2), replaced_token_label.view(-1))
            next_sentence_loss = ce_loss_fct(seq_relationship_score.view(-1, 2), batch["align_label"].view(-1))
            tri_label_loss = ce_loss_fct(tri_label_score.view(-1, 3), batch["tri_label"].view(-1))
            reg_label_loss = self.mse_loss(reg_label_score.view(-1), batch["reg_label"].view(-1))

            total_loss.append(10.0 * hallu_detect_loss if not torch.isnan(hallu_detect_loss).item() else 0.0)
            total_loss.append(
                0.2 * masked_lm_loss if (not torch.isnan(masked_lm_loss).item() and self.need_mlm) else 0.0
            )
            total_loss.append(next_sentence_loss if not torch.isnan(next_sentence_loss).item() else 0.0)
            total_loss.append(tri_label_loss if not torch.isnan(tri_label_loss).item() else 0.0)
            total_loss.append(reg_label_loss if not torch.isnan(reg_label_loss).item() else 0.0)

            total_loss = sum(total_loss)

        return ModelOutput(
            loss=total_loss,
            all_loss=(
                [masked_lm_loss, next_sentence_loss, tri_label_loss, reg_label_loss, hallu_detect_loss]
                if "mlm_label" in batch.keys()
                else None
            ),
            prediction_logits=hallu_detect_score,
            seq_relationship_logits=seq_relationship_score,
            tri_label_logits=tri_label_score,
            reg_label_logits=reg_label_score,
            hidden_states=base_model_output.hidden_states,
            attentions=base_model_output.attentions,
        )


class ElectraDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 2)
        self.config = config
        self.gelu = nn.GELU()

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = self.gelu(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)

        return logits


@dataclass
class ModelOutput:
    loss: Optional[torch.FloatTensor] = None
    all_loss: Optional[list] = None
    loss_nums: Optional[list] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    tri_label_logits: torch.FloatTensor = None
    reg_label_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


if "__main__" == __name__:
    scorer = AlignScore(
        model="roberta-base",
        batch_size=1,
        device="cuda:0",
        ckpt_path="models/AlignScore-large.ckpt",
        evaluation_mode="bin_sp",
    )
    score = scorer.score(
        contexts=[
            'Title: Planet of the Apes (1968 film)\nchimpanzees: animal psychologist Zira (Kim Hunter) and surgeon Galen (Wright King). While unable to speak as his throat wound is healing, called "Bright Eyes" by Zira and placed with one of the captive primitive humans he later names "Nova", Taylor observes the enhanced society of talking apes and in a strict caste system: the gorillas being the military police, hunters and workers; the orangutans overseeing the affairs of government, science, and religion; and intellectual chimpanzees being mostly scientists. While their society is a theocracy similar to the beginnings of the human Industrial Era, the apes consider the primitive humans as'
        ],
        claims=["In the 1968 film Planet of the Apes, Galen was played by Wright King."],
    )