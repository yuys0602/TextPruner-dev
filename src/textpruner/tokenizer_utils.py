import torch
from torch import nn
import os
from itertools import chain
from collections import Counter
from tqdm import tqdm
import logging
import json
logger = logging.getLogger(__name__)
try:
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
except ImportError:
    logger.warning("Could not import sentencepiece. Pruning embeddings of sentencepiece-based model is not available.")


def count_frequency(self, texts_gen):
    token_counter = Counter()

    for text in texts_gen:
        tokens = self.tokenizer.tokenize(text)
        token_counter.update(tokens)

    all_tokens = [k for (k, v) in token_counter.most_common()]
    all_token_indices = self.tokenizer.convert_tokens_to_ids(all_tokens)

    return all_tokens, all_token_indices


class Tokenize:
    @classmethod
    def tokenize(cls, dataiter, tokenizer, fn=None) -> Counter :
        assert not isinstance(dataiter,str), "dataiter is assumed to be a collection (list, tuple, ...) of strings, not a single string"
        token_ids = Counter()
        for item in tqdm(dataiter):
            if fn is not None:
                item = fn(item) # pre-transform
            if isinstance(item, str):
                token_ids.update(tokenizer.encode(item, add_special_tokens=True))
            else:
                assert isinstance(item[0],str) # list of string
                token_ids.update(list(chain(*(tokenizer.encode(i, add_special_tokens=True) for i in item))))
        return token_ids


class SubwordTokenizer:
    @staticmethod
    def get_token_ids(tokenizer, dataiter=None, additional_tokens=None, additional_token_ids=None, min_count=1):
        token_ids = []
        # add special tokens
        special_token_ids = list(tokenizer.all_special_ids)

        normal_token_ids = []
        if dataiter is not None:
            token_ids_counter = Tokenize.tokenize(dataiter, tokenizer)
            normal_token_ids += [k for k,v in token_ids_counter.items() if v >= min_count]
        if additional_tokens is not None and len(additional_tokens) > 0:
            normal_token_ids += list(
                tokenizer.convert_tokens_to_ids(additional_tokens))
        if additional_token_ids is not None and len(additional_token_ids) > 0:
            normal_token_ids += list(additional_token_ids)
        normal_token_ids = list(set(normal_token_ids)-set(special_token_ids))
        token_ids = sorted(special_token_ids + normal_token_ids)
        return token_ids

    @staticmethod
    def save_vocab(tokenizer, token_ids, outdir):
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        pruned_vocab_file = os.path.join(outdir, 'vocab.txt')
        with open(pruned_vocab_file, 'w', encoding='utf-8') as f:
            for token in tokens:
                f.write(token+'\n')
        print(f"New embedding size {len(token_ids)} pruned vocab file has been saved to {pruned_vocab_file}. Reintialize the tokenizer!")



class XLMRSentencepieceTokenizer:

    @staticmethod
    def get_token_ids(tokenizer, dataiter=None, additional_tokens=None, additional_token_ids=None, min_count=1):
        token_ids = []
        # add special tokens
        # should equal to [0,1,2,3,size +1]
        special_token_ids = list(tokenizer.all_special_ids)

        normal_token_ids = []
        if dataiter is not None:
            token_ids_counter = Tokenize.tokenize(dataiter, tokenizer)
            normal_token_ids += [k for k,v in token_ids_counter.items() if v >= min_count]
        if additional_tokens is not None and len(additional_tokens) > 0:
            normal_token_ids += list(
                tokenizer.convert_tokens_to_ids(additional_tokens))
        if additional_token_ids is not None and len(additional_token_ids) > 0:
            normal_token_ids += list(additional_token_ids)
        normal_token_ids = list(set(normal_token_ids)-set(special_token_ids))
        token_ids = sorted(special_token_ids + normal_token_ids) # to make sure [0,1,2,3, ...., <mask>]
        return token_ids

    @staticmethod
    def save_vocab(tokenizer, token_ids, outdir):
        fairseq_offset = 1
        # {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
        fairseq_special_tokens_ids = [0, 1, 2, 3]
        fairseq_special_tokens_ids.append(
            len(tokenizer.sp_model) + fairseq_offset)  # ["<mask>"]
        # remove special tokens
        token_ids = [
            t for t in token_ids if t not in fairseq_special_tokens_ids]

        # special tokens + normal tokens
        spm_token_ids = [0, 1, 2] + \
            [t-fairseq_offset for t in token_ids]
        assert len(spm_token_ids) == len(set(spm_token_ids))


        m = sp_pb2_model.ModelProto()
        m.ParseFromString(tokenizer.sp_model.serialized_model_proto())

        spm_tokens = set([m.pieces[i].piece for i in spm_token_ids])
        new_pieces = [p for p in m.pieces if p.piece in spm_tokens]

        # delete all
        del m.pieces[:]
        m.pieces.extend(new_pieces)

        # #debug
        # #debug
        # print ("spm_token_ids:",spm_token_ids)
        # print ("spm_tokens:",spm_tokens)
        # print ('new pieces:',[p.piece for p in m.pieces])

        pruned_vocab_file = os.path.join(outdir, 'sentencepiece.bpe.model')
        with open(pruned_vocab_file, 'wb') as f:
            f.write(m.SerializeToString())
        print(f"New embedding size {len(new_pieces)+2} pruned vocab file has been saved to {pruned_vocab_file}. Reintialize the tokenizer!")



class RobertaGPT2Tokenizer:

    @staticmethod
    def get_token_ids(tokenizer, dataiter=None, additional_tokens=None, additional_token_ids=None, min_count=1):
        token_ids = []
        # add special tokens
        special_token_ids = [0, 1, 2, 3]
        special_token_ids += [len(tokenizer)-4+i for i in range(4)]  # ["unusedword0000","unusedword0001","unusedword0002","<mask>"]
        # remove special tokens, special tokens + normal tokens

        normal_token_ids = []
        if dataiter is not None:
            token_ids_counter = Tokenize.tokenize(dataiter, tokenizer)
            normal_token_ids += [k for k,v in token_ids_counter.items() if v >= min_count]
        if additional_tokens is not None and len(additional_tokens) > 0:
            normal_token_ids += list(
                tokenizer.convert_tokens_to_ids(additional_tokens))
        if additional_token_ids is not None and len(additional_token_ids) > 0:
            normal_token_ids += list(additional_token_ids)
        normal_token_ids = list(set(normal_token_ids)-set(special_token_ids))
        token_ids = sorted(special_token_ids + normal_token_ids) # to make sure [0,1,2,3, ...., <mask>]
        return token_ids

    @staticmethod
    def save_vocab(tokenizer, token_ids, outdir):
        
        assert len(token_ids) == len(set(token_ids))

        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        token_dict = {}
        for i in range(len(tokens)):
            token_dict[tokens[i]] = i

        pruned_vocab_file = os.path.join(outdir, 'vocab.json')
        with open(pruned_vocab_file, 'w', encoding='utf-8') as f:
            json.dump(token_dict, f)
        print(f"New embedding size {len(token_ids)} pruned vocab file has been saved to {pruned_vocab_file}. Reintialize the tokenizer!")

        index = 0
        bpe_ranks = sorted(tokenizer.bpe_ranks.items(), key = lambda k: k[1])
        pruned_merges_file = os.path.join(outdir, 'merges.txt')
        with open(pruned_merges_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, _ in bpe_ranks:
                writer.write(bpe_tokens[0] + " " + bpe_tokens[1] + "\n")
