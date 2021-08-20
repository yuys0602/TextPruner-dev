import torch
import logging
from .tokenizer_utils import SubwordTokenizer, XLMRSentencepieceTokenizer, RobertaGPT2Tokenizer
from .model_utils import bert_set_embeddings, xlmr_set_embeddings, roberta_set_embeddings
logger = logging.getLogger(__name__)
MODEL_MAP = {
    'bert':
    {
        'get_token_ids': SubwordTokenizer.get_token_ids,
        'save_vocab': SubwordTokenizer.save_vocab,
        'set_embeddings': bert_set_embeddings
    },
    'xlm-roberta':
    {
        'get_token_ids': XLMRSentencepieceTokenizer.get_token_ids,
        'save_vocab': XLMRSentencepieceTokenizer.save_vocab,
        'set_embeddings': xlmr_set_embeddings
    },
    'roberta':
    {
        'get_token_ids':RobertaGPT2Tokenizer.get_token_ids,
        'save_vocab': RobertaGPT2Tokenizer.save_vocab,
        'set_embeddings': roberta_set_embeddings
    },
    'albert':
    {
        'get_token_ids': SubwordTokenizer.get_token_ids,
        'save_vocab': SubwordTokenizer.save_vocab,
        'set_embeddings': bert_set_embeddings
    },
    'electra':
    {
        'get_token_ids': SubwordTokenizer.get_token_ids,
        'save_vocab': SubwordTokenizer.save_vocab,
        'set_embeddings': bert_set_embeddings
    },
}

class GeneralConfig:
    def __init__(self, device : str = 'auto', output_dir: str = './pruned_models') -> None:
        self.output_dir = output_dir
        if device == 'auto':
            if torch.cuda.is_available():
                logger.info(f"Using current cuda device")
                self.device = torch.device('cuda')
            else:
                logger.info(f"Using cpu device")
                self.device = torch.device('cpu')
        else:
            self.device = device

class EmbeddingPruningConfig:
    def __init__(self, min_count = 1) -> None:
        self.min_count = min_count

class TransformerPruningConfig:
    def __init__(self, 
                ffn_size : int,
                num_of_heads: int,
                batch_size : int = 32,
                is_iterative : bool = False,
                ffn_is_layerwise : bool = True,
                head_is_layerwise : bool = True,
                n_iter : int = 4
                ) -> None:
        self.is_iterative = is_iterative
        self.ffn_is_layerwise = ffn_is_layerwise
        self.head_is_layerwise = head_is_layerwise
        self.ffn_size = ffn_size
        self.num_of_heads = num_of_heads
        self.batch_size = batch_size
        self.n_iter = n_iter