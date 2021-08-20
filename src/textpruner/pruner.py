import torch
from torch import nn
import numpy as np
import os
from .configurations import MODEL_MAP, TransformerPruningConfig, EmbeddingPruningConfig, GeneralConfig
from .utils import move_to_device, generate_mask
import logging
from tqdm import tqdm
from collections import abc
from .embedding_pruner import EmbeddingPruner
logger = logging.getLogger(__name__)



class Pruner:
    def __init__(self, model_type, model=None, tokenizer=None,  outdir='./pruned-model'):
        assert not (
            model is None and tokenizer is None), "Nothing to do: model and tokenizer are both None"

        self.model_type = model_type
        self.model = model
        self.tokenizer = tokenizer
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

        self.embedding_pruner = EmbeddingPruner(model_type, self.model, tokenizer, outdir)

    def prune(self, dataiter):
        pass

    def prune_embeddings(self, dataiter=None, additional_tokens=None, additional_token_ids=None, min_count=1,save_model=False):

        token_ids = self.embedding_pruner.prune_embeddings(
            dataiter, additional_tokens, additional_token_ids, min_count, save_model=save_model)

        return token_ids