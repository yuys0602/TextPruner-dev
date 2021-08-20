import torch
from torch import nn
import numpy as np
import os
from .configurations import MODEL_MAP, EmbeddingPruningConfig, GeneralConfig
import logging
from tqdm import tqdm
from collections import abc
from typing import Optional
logger = logging.getLogger(__name__)

class EmbeddingPruner:
    def __init__(self, model, tokenizer,
                    general_config : Optional[GeneralConfig] = None,
                    embedding_pruning_config : Optional[EmbeddingPruningConfig] = None,
                    base_model_prefix : Optional[str] = None
):

        self.model = model
        if base_model_prefix is not None:
            self.base_model = getattr(model, base_model_prefix, model)
            self.model_type = self.base_model.config.model_type
        else:
            if hasattr(model, 'base_model_prefix'):
                self.base_model = getattr(model, model.base_model_prefix, model)
                if hasattr(self.base_model, 'config'):
                    self.model_type = self.base_model.config.model_type
                else:
                    raise ValueError("Cannot infer/get model_type! Maybe you should provide base_model_prefix")
            else:
                raise ValueError("Cannot infer/get model_type!") 
        assert self.model_type in MODEL_MAP, \
            f"Model type {self.model_type} is not supported, or not understood. Model type must be one of {list(MODEL_MAP.keys())}"


        self.tokenizer = tokenizer

        self.general_config = GeneralConfig() if general_config is None else general_config
        self.embedding_pruning_config = EmbeddingPruningConfig() if embedding_pruning_config is None else embedding_pruning_config


        self.get_token_ids = MODEL_MAP[self.model_type]['get_token_ids']
        self.set_embeddings = MODEL_MAP[self.model_type]['set_embeddings']
        self.save_vocab = MODEL_MAP[self.model_type]['save_vocab']

        self.new_token_ids = []

        os.makedirs(self.general_config.output_dir, exist_ok=True)

    def save_model(self):

        vocab_size = len(self.new_token_ids)
        self.base_model.config.vocab_size = vocab_size

        output_dir = os.path.join(self.general_config.output_dir, f'pruned_{vocab_size}V')
        os.makedirs(output_dir, exist_ok=True)

        self.save_vocab(self.tokenizer, self.new_token_ids, output_dir)
        if torch.__version__ >= '1.6':
            torch.save(self.model.state_dict(),os.path.join(output_dir,f'model.pkl'),_use_new_zipfile_serialization=False)
        else:
            torch.save(self.model.state_dict(),os.path.join(output_dir,f'model.pkl'))
        # save config
        config_dir = os.path.join(output_dir)
        self.base_model.config.save_pretrained(config_dir)
        logger.info(f"Model and configuration have been saved to {output_dir}")

    def prune_embeddings(self, dataiter=None, additional_tokens=None, 
                               additional_token_ids=None, save_model=False):
        min_count = self.embedding_pruning_config.min_count
        new_token_ids = self.get_token_ids(tokenizer=self.tokenizer,
                                       dataiter=dataiter,
                                       additional_tokens=additional_tokens,
                                       additional_token_ids=additional_token_ids,
                                       min_count=min_count)
        self.set_embeddings(model=self.base_model, token_ids=new_token_ids)
        self.new_token_ids = new_token_ids

        if save_model is True:
            self.save_model()
