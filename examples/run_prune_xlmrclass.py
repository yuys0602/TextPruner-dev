
from transformers import XLMRobertaTokenizer, XLMRobertaConfig, XLMRobertaModel
from textpruner import Pruner
import torch
import os
import json
import tqdm

from modeling import XLMRForGLUESimple


xlmr_vocab_file='./pretrained-models/xlm-r-base/sentencepiece.bpe.model'
xlmr_config_file='./pretrained-models/xlm-r-base/config.json'
xlmr_class_ckpt_file='/work/rc/zqyang5/cross-lingual/xnli/xlmr/xnli_XbTrainEn_lr3e4_s4_bs32/gs42948.pkl'

def init_xlmr_model(vocab_file=None, config_file=None, ckpt_file=None):
    config=XLMRobertaConfig.from_json_file(config_file) if config_file is not None else None
    tokenizer=XLMRobertaTokenizer(vocab_file=vocab_file) if vocab_file is not None else None
    model = XLMRobertaModel(config=config) if config is not None else None
    return tokenizer,model


def init_xlmr_class_model(vocab_file=None, config_file=None, ckpt_file=None):
    tokenizer=XLMRobertaTokenizer(vocab_file=vocab_file) if vocab_file is not None else None
    if config_file is not None:
        config=XLMRobertaConfig.from_json_file(config_file)
        model = XLMRForGLUESimple(config=config, num_labels=3)
        if ckpt_file is not None:
            sd = torch.load(ckpt_file,map_location='cpu')
            a1, a2 = model.load_state_dict(sd,strict=False)
            print (a1)
            print (a2)
    else:
        config = None
        model = None
    return tokenizer,model


def extract_sentences_from_xnli(data_files):
    results = []
    for data_file in data_files:
        with open(data_file,'r',encoding='utf-8') as f:
            lines = f.readlines()
        for line in tqdm.tqdm(lines):
                fields = line.strip().split('\t')
                for field in fields:
                    results.append(field)
    return results

if __name__ == '__main__':
    '''
    data_files = ['./datasets/multinli.train.en.tsv','./datasets/multinli.train.zh.tsv']
    lines = extract_sentences_from_xnli(data_files)
    print("Number of lines: ", len(lines))
    from textpruner.tokenizer_utils import Tokenize
    token_ids_counter = Tokenize.tokenize(lines,tokenizer)
    torch.save(token_ids_counter,'datasets/xnli-token-counter')
    '''

    token_ids_counter = torch.load('datasets/xnli-token-counter')
    min_count = 1
    token_ids = [k for k,v in token_ids_counter.items() if v >=min_count]
    print ("Number of different tokens in XNLI (zh,en):", len(token_ids))
    
    tokenizer, model = init_xlmr_class_model(xlmr_vocab_file, xlmr_config_file, xlmr_class_ckpt_file)
    print("Current Vocab size:",tokenizer.vocab_size)
    print("Current Embedding size:", model.roberta.get_input_embeddings().weight.shape)


    outdir='./pruned-models/'
    pruner = Pruner('xlmr',model.roberta,tokenizer,outdir=outdir)
    token_ids = pruner.prune_embeddings(additional_token_ids=token_ids)

    print("New embedding size:", model.roberta.get_input_embeddings().weight.shape)
    new_tokenizer, _ = init_xlmr_model(outdir+'/sentencepiece.bpe.model')
    print("New vocab size:", new_tokenizer.vocab_size)


