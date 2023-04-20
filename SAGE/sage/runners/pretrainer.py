"""
Pretrainer class for Explainer model

Copyright (c) 2022 Hocheol Lim.
"""

from typing import List

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sage.data.char_dict import SmilesCharDictionary
from sage.logger.abstract_logger import AbstractLogger
from sage.models.handlers import AbstractGeneratorHandler
from sage.utils.smiles import smiles_to_actions

class PreTrainer:
    def __init__(
        self,
        char_dict: SmilesCharDictionary,
        dataset: List[str],
        generator_handler: AbstractGeneratorHandler,
        num_epochs: int,
        batch_size: int,
        save_dir: str,
        num_workers: int,
        device: torch.device,
        logger: AbstractLogger,
    ):
        self.generator_handler = generator_handler
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.device = device
        self.logger = logger

        action_dataset, _ = smiles_to_actions(char_dict=char_dict, smis=dataset)
        action_dataset_ten = TensorDataset(torch.LongTensor(action_dataset))  # type: ignore
        self.dataset_loader: DataLoader = DataLoader(
            dataset=action_dataset_ten,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    def pretrain(self):
        for epoch in tqdm(range(self.num_epochs)):
            for actions in self.dataset_loader:
                loss = self.generator_handler.train_on_action_batch(
                    actions=actions[0], device=self.device
                )
                self.logger.log_metric("loss", loss)
            self.generator_handler.save(self.save_dir)
