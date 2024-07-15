import json
import os

import numpy as np

from src.utils.util import ROOT_DIR


class TemplateFormatter(object):
    def __init__(
        self,
        dataset,
        setting_name,
        num_passages=5,
    ):
        self.data_path = os.path.join(ROOT_DIR, "data", "prompts", "{}_{}.json".format(dataset, setting_name))

        self.num_passages = num_passages

        self.templates = json.load(open(self.data_path))
        # Single instrution at the moment. Really just a single template we consider right now.
        self.prompt = self.templates["demo_prompt"]
        self.doc_prompt = self.templates["doc_prompt"]
        self.demo_sep = self.templates["demo_sep"]
        self.demos = self.templates["demos"]
        self.instruction = self.templates["instruction"]

    def _get_doc_text_individual(self, doc, doc_id, attribution_representation):
        doc_prompt = self.doc_prompt
        text = doc["text"]
        if attribution_representation == "num":
            doc_prompt.replace("{T}", doc["title"]).replace("{P}", text).replace("{ID}", str(doc_id + 1))
        return doc_prompt

    def apply_templates_to_sample_append_query(self, question, docs, attribution_representation="num"):
        prompts = []
        prompt = prompt.replace("{INST}", self.instruction).replace("{Q}", question)

        for doc_id, doc in enumerate(docs[self.num_passages]):
            prompt_cp = prompt
            if "{D}" in prompt_cp:
                if self.num_docs == 0:
                    prompt_cp = prompt_cp.replace("{D}\n", "")  # if there is no doc we also delete the empty line
                else:
                    text = "".join(self._get_doc_text_individual(doc, doc_id + 1, attribution_representation))
                    prompt_cp = prompt_cp.replace("{D}", text)
                    doc_id += 1
                prompts.append(prompt_cp)

        return [prompts]

    def _get_doc_texts(self, docs, attribution_representation):
        doc_list = []
        doc_id = 0
        for doc in docs[: self.num_passages]:
            # print("DOC", doc)
            doc_prompt = self.doc_prompt
            text = doc["text"]
            if attribution_representation == "num":
                doc_prompt = (
                    doc_prompt.replace("{T}", doc["title"]).replace("{P}", text).replace("{ID}", str(doc_id + 1))
                )
            doc_list.append(doc_prompt)
            doc_id += 1
        return doc_list

    def apply_templates_to_samples(
        self, questions, docs, answers=None, attribution_representation="num", use_chat_template=False
    ):
        prompts = []
        for i, question in enumerate(questions):
            answer = answers[i] if answers else None
            prompt = ""
            prompt = self.apply_templates_to_sample(
                question, docs[i], answer, attribution_representation, use_chat_template=use_chat_template
            )
            prompts.append(prompt)
        return prompts

    def apply_templates_to_samples_in_context(
        self,
        questions,
        docs,
        num_training_samples,
        answers=None,
        attribution_representation="num",
        query_id=None,
        use_chat_template=False,
    ):
        prompts = []
        for i, question in enumerate(questions):
            prompt = ""

            if query_id:
                demos_filtered = [
                    j for j in range(len(self.demos)) if str(j + 1) != query_id[i]
                ]  # Do not use a in-context demo for a prediction demo
            else:
                demos_filtered = len(self.demos)
            train_ids = np.random.choice(demos_filtered, num_training_samples, replace=False)
            for train_id in train_ids:
                train_item = self.demos[train_id]
                prompt += self.apply_templates_to_sample(
                    train_item["question"],
                    train_item["docs"],
                    train_item["answer"],
                    attribution_representation,
                    use_chat_template=False,
                )  # always insert answer even when using chat temmpalte
                prompt += self.demo_sep

            prompt = prompt + self.apply_templates_to_sample(
                question,
                docs[i],
                answers=answers[i] if answers else None,
                attribution_representation=attribution_representation,
                use_chat_template=use_chat_template,
            )  # answer is added after chat template rendering, so not added if set true
            prompts.append(prompt)
        return prompts

    def apply_templates_to_sample(
        self, question, docs, answers=None, attribution_representation="num", use_chat_template=False
    ):
        prompt = self.prompt.replace("{INST}", self.instruction).replace("{Q}", question)

        if "{D}" in prompt:
            if self.num_passages == 0:
                prompt = prompt.replace("{D}\n", "")  # if there is no doc we also delete the empty line
            else:
                text = "".join(self._get_doc_texts(docs, attribution_representation))
                prompt = prompt.replace("{D}", text)

        if (not answers) or use_chat_template:
            prompt = prompt.replace("{A}", "")
        else:
            prompt = prompt.replace("{A}", "").rstrip() + answers  # + "\n\n\n"

        return prompt
