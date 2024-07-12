import copy
import json
import logging
import math
import os
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import CrossEntropyLoss, NLLLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.constants import HF_ACCESS_TOKEN
from src.utils.util import get_max_memory

logger = logging.getLogger(__name__)


class DummyLayer(nn.Module):
    """
    DummyLayer to ensure that the gradient checkpointing will assign output layer as require_grad=True.
    Reference: https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
    """

    def __init__(self):
        super().__init__()
        self.dummy_bias = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def forward(self, input_ids, labels, attention_mask, use_cache=False):
        return input_ids + self.dummy_bias.to(input_ids) - self.dummy_bias.to(input_ids)


class CastOutputToFloat(nn.Sequential):
    """
    Used for int8 training of model
    """

    def forward(self, x):
        return super().forward(x).to(torch.float32)


class LLM(nn.Module):
    """
    Encoder Decoder
    """

    def __init__(self, config, rank):
        """
        :param config
        """
        super().__init__()
        self.config = config
        int8 = True

        start_time = time.time()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.reader_model_origin,
            device_map="auto",
            max_memory=get_max_memory(rank=rank),
            offload_folder="offload/{}".format(self.config.reader_model_origin),
            load_in_8bit=int8,
            token=HF_ACCESS_TOKEN,
            trust_remote_code=True,  # for DeciLM
        )

        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

        self.model.gradient_checkpointing_enable()  # reduce number of stored activations
        self.model.enable_input_require_grads()

        self.model.lm_head = CastOutputToFloat(self.model.lm_head)

        config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.reader_model_origin, token=HF_ACCESS_TOKEN
        )  # , use_fast=False)

        logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def forward(self, importance_scores, **kwargs):
        """
        Predict the lbl for particular pet
        :param batch:
        :param pet:
        :return:
        """
        if self.config.focused_learning and "soft_masking" in self.config.focused_learning_mode:
            labels = copy.deepcopy(kwargs["labels"])
            kwargs["labels"] = None
            outputs = self.model.model.model(  # Is it wrapped around model, hence three times, jeeez...
                input_ids=kwargs["input_ids"],
                attention_mask=kwargs["attention_mask"],
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                use_cache=None,
                output_attentions=self.model.config.output_attentions,
                output_hidden_states=self.model.config.output_hidden_states,
                return_dict=self.model.config.use_return_dict,
            )
            hidden_states = outputs[0]
            logits = self.model.lm_head(hidden_states)
            logits = logits.float()

            weights = torch.tensor([1.0] * logits.size(-1)).to(self.model.device)

            if self.config.focused_learning_mode == "soft_masking_class_rescaling":
                # rescaling = torch.tensor([x[0] for x in importance_scores[0]]).to(self.model.device)
                rescaling_dict = {x[1]: x[0] for x in importance_scores[0]}
                weights = []
                for i in range(logits.size(-1)):
                    if i in rescaling_dict:
                        weights.append(rescaling_dict[i])
                    else:
                        weights.append(1.0)
                weights = torch.tensor(weights).to(self.model.device)
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(weight=weights)
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                output = (logits,) + outputs[1:]
                output = (loss,) + output
            elif "soft_masking_token_rescaling" in self.config.focused_learning_mode:
                rescaling = [x[0] for x in importance_scores[0]]
                if "length_bias" in self.config.focused_learning_mode:
                    tokens = [x[1] for x in importance_scores[0] if x[1] >= 0]  # not mask token
                    rescaling_tokens = [
                        (i, self.tokenizer.decode([x[1]])) if x[1] >= 0 else (i, -100)
                        for i, x in enumerate(importance_scores[0])
                    ]
                    sentence_weight = []
                    curr_sent = 1
                    reweight_func = lambda x: 1 * math.exp(-self.config.length_bias_term_factor * x)
                    for tok in rescaling_tokens:
                        if tok[1] == -100:
                            sentence_weight.append(1)
                            continue
                        weight = reweight_func(curr_sent)
                        sentence_weight.append(weight)
                        if "." in tok[1]:
                            curr_sent += 1
                    rescaling = [x * sentence_weight[i] for i, x in enumerate(rescaling)]
                rescaling = torch.tensor([x[0] for x in importance_scores[0]]).to(self.model.device)
                rescaling = torch.unsqueeze(rescaling, dim=0)
                rescaling = torch.unsqueeze(rescaling, dim=-1)
                rescaling = rescaling.expand(rescaling.size(0), rescaling.size(1), logits.size(-1))

                shift_rescaling = rescaling[..., 1:, :].contiguous()

                logits = self.log_softmax(logits)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_logits = shift_logits * shift_rescaling
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = NLLLoss()
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)

                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                output = (logits,) + outputs[1:]
                output = (loss,) + output
            else:
                logger.error("Soft masking method {} not recognized".format(self.config.focused_learning_mode))
                sys.exit()
        else:
            output = self.model(**kwargs)
        return output  # element 0 in list should be loss
