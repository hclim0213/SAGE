"""
Loading methods to ease object instantiation

Copyright (c) 2022 Hocheol Lim.
"""

import logging
from typing import List, Optional, Union

import torch

from sage.logger import CommandLineLogger, NeptuneLogger
from sage.models.apprentice import LSTMGenerator
from sage.models.handlers import (
    GeneticOperatorHandler,
    LSTMGeneratorHandler,
)


def load_logger(args, tags=None):
    if args.logger_type == "Neptune":
        logger = NeptuneLogger(args, tags)
    elif args.logger_type == "CommandLine":
        logger = CommandLineLogger(args)
    else:
        raise NotImplementedError

    return logger


def load_neural_apprentice(args):
    if args.model_type == "LSTM":
        neural_apprentice = LSTMGenerator.load(load_dir=args.apprentice_load_dir)
    else:
        raise ValueError(f"{args.model_type} is not a valid model-type")

    return neural_apprentice


def load_apprentice_handler(model, optimizer, char_dict, max_sampling_batch_size, args):
    if args.model_type == "LSTM":
        apprentice_handler = LSTMGeneratorHandler(
            model=model,
            optimizer=optimizer,
            char_dict=char_dict,
            max_sampling_batch_size=max_sampling_batch_size,
        )
    
    return apprentice_handler


def load_genetic_experts(
    expert_types: List[str],
    args,
) -> List[GeneticOperatorHandler]:
    experts = []
    for ge_type in expert_types:
        expert_handler = GeneticOperatorHandler(
            crossover_type=ge_type,
            mutation_type=ge_type,
            mutation_initial_rate=args.mutation_initial_rate,
        )
        experts.append(expert_handler)
    return experts


def load_generator(input_size: int, args):
    if args.model_type == "LSTM":
        generator = LSTMGenerator(
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_size=input_size,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
    
    return generator

