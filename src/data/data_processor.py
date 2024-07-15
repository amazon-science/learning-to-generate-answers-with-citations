import json
import logging
import os
import sys

from src.utils.util import ROOT_DIR

logger = logging.getLogger(__name__)


class DatasetProcessor(object):
    def __init__(self, split, config):
        self.split = split
        self.dataset = config.dataset
        self.num_samples = config.num_samples
        self.use_gold_docs = config.use_gold_passages
        self.use_file_docs = config.use_file_passages
        self.num_retrieved_docs = max(config.retriever_n_context, config.n_context)
        self.num_retrieved_docs_max = max(
            100, self.num_retrieved_docs
        )  # To enable using saved files with larger number of retrieved docs, efficiency
        self.overwrite_data = config.overwrite_data
        self.is_debug = config.is_debug
        self.unlabeled_samples_cutoff = config.unlabeled_samples_cutoff

        exp_dir = config.exp_dir
        retriever_name = config.retriever
        seed = config.seed
        ws_attribution_training = config.ws_attribution_training
        ws_iteration = config.ws_iteration

        self.questions = {}
        self.answers = {}
        self.passage_ids = {}
        self.passages = {}
        self.gold_passages = {}
        self.qa_pairs = {}
        self.annotations = {}
        self.claims = {}

        curr_samples = 0
        if split == "train" and ws_iteration <= 1:
            save_path = os.path.join(
                ROOT_DIR,
                "data",
                self.dataset,
                "{}_{}_{}_top{}_alce.json".format(self.dataset, self.split, retriever_name, self.num_retrieved_docs),
            )
            save_path_max = os.path.join(
                ROOT_DIR,
                "data",
                self.dataset,
                "{}_{}_{}_top{}_alce.json".format(self.dataset, split, retriever_name, self.num_retrieved_docs_max),
            )
            base_path = os.path.join(
                ROOT_DIR, "data", self.dataset, "{}_{}_alce.json".format(self.dataset, self.split)
            )  # TODO: Fine while doing ALCE.
        elif split == "train" and ws_iteration > 1:
            save_path_0 = os.path.join(
                ROOT_DIR,
                "data",
                self.dataset,
                "{}_{}_{}_top{}_alce.json".format(self.dataset, self.split, retriever_name, self.num_retrieved_docs),
            )
            save_path_max_0 = os.path.join(
                ROOT_DIR,
                "data",
                self.dataset,
                "{}_{}_{}_top{}_alce.json".format(
                    self.dataset, self.split, retriever_name, self.num_retrieved_docs_max
                ),
            )
            base_path_0 = os.path.join(
                ROOT_DIR, "data", self.dataset, "{}_{}_alce.json".format(self.dataset, self.split)
            )  # TODO: Fine while doing ALCE.
            self.load_data(save_path_0, save_path_max_0, base_path_0)
            exp_dir_last_iteration = exp_dir.replace(
                "iter_{}".format(ws_iteration), "iter_{}".format(ws_iteration - 1)
            )
            assert os.path.isdir(
                exp_dir_last_iteration
            ), "Experiment path from last iteration for weakly supervised attribution training does not exist: {}".format(
                exp_dir_last_iteration
            )
            save_path = os.path.join(exp_dir_last_iteration, "test_pred_filtered.jsonl")
            save_path_max = None
            base_path = None
            curr_samples = len(self.questions)
        elif split == "test":
            save_path = os.path.join(
                ROOT_DIR,
                "data",
                self.dataset,
                "{}_{}_{}_top{}.json".format(self.dataset, "train", retriever_name, self.num_retrieved_docs),
            )
            save_path_max = os.path.join(
                ROOT_DIR,
                "data",
                self.dataset,
                "{}_{}_{}_top{}.json".format(self.dataset, "train", retriever_name, self.num_retrieved_docs_max),
            )
            base_path = os.path.join(
                ROOT_DIR, "data", self.dataset, "{}_{}.json".format(self.dataset, self.split)
            )  # TODO:
        else:
            save_path = os.path.join(
                ROOT_DIR,
                "data",
                self.dataset,
                "{}_{}_{}_top{}.json".format(
                    self.dataset, self.split, retriever_name, self.num_retrieved_docs, self.use_gold_docs
                ),
            )
            save_path_max = os.path.join(
                ROOT_DIR,
                "data",
                self.dataset,
                "{}_{}_{}_top{}.json".format(
                    self.dataset, self.split, retriever_name, self.num_retrieved_docs_max, self.use_gold_docs
                ),
            )
            base_path = os.path.join(
                ROOT_DIR, "data", self.dataset, "{}_{}.json".format(self.dataset, self.split)
            )  # TODO: Fine while doing ALCE.

        self.load_data(save_path, save_path_max, base_path)

        if (
            split == "train" and ws_iteration > 1 and len(self.questions) < curr_samples + 3
        ):  # If less than 5 samples were added through weak supervision, abort.
            logger.warning(
                "Only a total of {} instances were added via weak supervision, this is not sufficient. Try lower filtering threshold...".format(
                    len(self.questions) - curr_samples
                )
            )
            # sys.exit()

    def load_data(self, save_path, save_path_max, base_path):
        logging.info("Trying to find data from {} ...".format(save_path))
        if os.path.exists(save_path) and not self.overwrite_data:
            logging.info("Load existing data from {} ...".format(save_path))
            self.load_existing_data(save_path, self.is_debug)
            return
        elif os.path.exists(save_path_max) and not self.overwrite_data:
            logging.info("Load existing data from MAX {} ...".format(save_path_max))
            self.load_existing_data(save_path_max, self.is_debug)
            return
        elif os.path.exists(base_path) and (self.use_gold_docs or not self.use_file_docs):
            logging.info("Load existing GOLD data from {} ...".format(base_path))
            self.load_existing_data(base_path, self.is_debug)
        else:
            logging.error(
                "Cannot find specified data path {} and also not base path {} or you want to use file docs.".format(
                    save_path, base_path
                )
            )
            sys.exit()

    def load_existing_data(self, save_path, is_debug):
        # ALCE training, evaluation, and unlabeled training data are formatted slightly differently.
        # TODO: Unify representation of data to simplify reading in the files.
        if self.split == "train":
            with open(save_path, "r") as f_in:
                lines = f_in.readlines()
                cutoff = self.num_samples if not is_debug else 5
                for i, line in enumerate(lines[:cutoff]):
                    sample = json.loads(line)
                    qid = sample["sample_id"] if "sample_id" in sample else str(i + 1)
                    self.questions[qid] = sample["question"]
                    self.answers[qid] = (
                        sample["answers"][0] if isinstance(sample["answers"], list) else sample["answers"]
                    )
                    self.passages[qid] = sample["passages"][: self.num_retrieved_docs]
                    self.claims[qid] = sample["claims"] if "claims" in sample else []
                    self.gold_passages[qid] = []
                    self.qa_pairs[qid] = sample["qa_pairs"] if "qa_pairs" in sample else []
                    self.annotations[qid] = sample["annotations"] if "annotations" in sample else []
        elif self.split == "test":
            with open(save_path, "r") as f_in:
                content = json.load(f_in)
                cutoff = self.unlabeled_samples_cutoff if not is_debug else 10
                for i, sample in enumerate(content[:cutoff]):
                    if self.dataset == "hagrid":
                        qid = "hagrid_" + str(i + 1)
                    else:
                        qid = sample["sample_id"]
                    self.questions[qid] = sample["question"]
                    if self.dataset == "hagrid":
                        self.answers[qid] = sample["answers"][0]
                    else:
                        self.answers[qid] = sample["annotations"][0]["long_answer"]
                    self.passages[qid] = sample["docs"][: self.num_retrieved_docs]
                    self.gold_passages[qid] = []
                    self.claims[qid] = sample["claims"] if "claims" in sample else []
                    self.qa_pairs[qid] = sample["qa_pairs"] if "qa_pairs" in sample else []
                    self.annotations[qid] = sample["annotations"] if "annotations" in sample else []
        else:
            with open(save_path, "r") as f_in:
                content = json.load(f_in)
                cutoff = len(content) + 1 if not is_debug else 10
                for i, sample in enumerate(content[:cutoff]):
                    if self.dataset == "eli5":
                        qid = "eli5_" + str(i + 1)
                    elif self.dataset == "factscore":
                        qid = "factscore_" + str(i + 1)
                    elif self.dataset == "hagrid":
                        qid = "hagrid_" + str(i + 1)
                    else:
                        qid = sample["sample_id"] if self.dataset != "qampari" else sample["id"]
                    self.questions[qid] = sample["question"]
                    if self.dataset == "qampari":
                        if isinstance(sample["answer"], str):
                            self.answers[qid] = [[x] for x in sample["answer"].rstrip(".").rstrip(",").split(", ")]
                        else:
                            self.answers[qid] = sample["answer"]
                    if self.dataset == "hagrid":
                        self.answers[qid] = sample["answers"][0]
                    else:
                        self.answers[qid] = sample["answer"]
                    self.passages[qid] = sample["docs"][: self.num_retrieved_docs]
                    self.gold_passages[qid] = []
                    self.claims[qid] = sample["claims"] if "claims" in sample else []
                    self.qa_pairs[qid] = sample["qa_pairs"] if "qa_pairs" in sample else []
                    self.annotations[qid] = sample["annotations"] if "annotations" in sample else []
        return

    def save_data(self):
        with open(self.save_path, "w") as f_out:
            samples = []
            for qid, question in self.questions.items():
                sample = {
                    "id": qid,
                    "question": question,
                    "answer": self.answers[qid],
                    "passages": self.passages[qid],
                    "annotations": {"knowledge": self.gold_passages[qid], "long_answer": self.answers[qid]},
                    "qa_pairs": self.qa_pairs["qid"],
                }  # TODO: a bit hacky with the long answer but should be fine for now.
                samples.append(sample)
            json.dump(sample, f_out)
            return
