import copy
import logging
import os
import pathlib
import random

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from tqdm import tqdm

from src.constants import (
    ANSWER_KEY,
    GOLD_PASSAGE_KEY,
    ID_KEY,
    ORIGINAL_ID_KEY,
    PASSAGE_ID_KEY,
    PASSAGE_TEXT_KEY,
    PASSAGE_TITLE_KEY,
    QA_PAIRS_KEY,
    QUESTION_KEY,
)
from src.data.data_processor import DatasetProcessor

logger = logging.getLogger(__name__)


class FinetuneDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def _process_dataset(self, dataset, is_inference=False):
        qid = 0
        processed_data = {}

        for key in tqdm(dataset.questions):
            question = dataset.questions[key]
            answer = dataset.answers[key]

            processed_data[qid] = {
                ID_KEY: qid,  # Unique id starting from 0, required by dataloader
                ORIGINAL_ID_KEY: key,
                QUESTION_KEY: question,  # Question
                # ANSWER_KEY: answer,
            }

            qid += 1
        return processed_data

    def get_num_training_samples(self):
        return len(
            DatasetProcessor(
                split="train",
                config=self.config,
            ).questions.keys()
        )

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        if stage == "fit":
            self.train_dataset = DatasetProcessor(
                split="train",
                config=self.config,
            )
            self.train_dataset_processed = self._process_dataset(self.train_dataset, is_inference=False)
            self.train_dataset = FinetuneDatasetWithTemplate(self.train_dataset_processed, self.train_dataset)
            logger.info(f"Train size {len(self.train_dataset)}")

            # Not needed but due to pytorch code execution needed.
            # TODO: Refactor later so that we do without the val/test situation for eval and data generation. Bad style.
            self.dev_dataset = DatasetProcessor(
                split="eval",
                config=self.config,
            )
            self.dev_dataset_processed = self._process_dataset(self.dev_dataset, is_inference=True)
            self.dev_dataset = FinetuneDatasetWithTemplate(self.dev_dataset_processed, self.dev_dataset)
            logger.info(f"Val size {len(self.dev_dataset)}")
        elif stage in ["validate"] and not hasattr(self, "dev_dataset"):
            self.dev_dataset = DatasetProcessor(
                split="eval",
                config=self.config,
            )
            self.dev_dataset_processed = self._process_dataset(self.dev_dataset, is_inference=True)
            self.dev_dataset = FinetuneDatasetWithTemplate(self.dev_dataset_processed, self.dev_dataset)
            logger.info(f"Val size {len(self.dev_dataset)}")

        elif stage in ["predict", "test"]:
            self.test_dataset = DatasetProcessor(
                split="test",
                config=self.config,
            )
            self.test_dataset_processed = self._process_dataset(self.test_dataset, is_inference=False)
            self.test_dataset = FinetuneDatasetWithTemplate(self.test_dataset_processed, self.test_dataset)
            logger.info(f"Eval size {len(self.test_dataset)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            # num_workers=torch.cuda.device_count(),
            num_workers=min([self.config.batch_size, self.config.num_workers]),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            # num_workers=torch.cuda.device_count(),
            num_workers=min([self.config.eval_batch_size, self.config.num_workers]),
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            # num_workers=torch.cuda.device_count(),
            num_workers=min([self.config.eval_batch_size, self.config.num_workers]),
        )


class FinetuneDatasetWithTemplate(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, dataset_original):
        super().__init__()
        self.dataset = dataset
        self.dataset_original = dataset_original

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        return self.dataset[key]
