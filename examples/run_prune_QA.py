
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import XLMRobertaTokenizer, XLMRobertaConfig, XLMRobertaModel
from textpruner import Pruner
import torch
import os
import json

from modeling import BertForQASimple


chinese_roberta_vocab_file='./pretrained-models/chinese-roberta-base/vocab.txt'
chinese_roberta_config_file='./pretrained-models/chinese-roberta-base/bert_config.json'

chinese_roberta_qa_ckpt_file='/work/rc/zqyang5/distill_chinese/cmrc2018/nodistill/cmrc2018BaseTrain_lr3e2_s4/gs508.pkl'

def init_bert_model(vocab_file=None, config_file=None, ckpt_file=None):
    config=BertConfig.from_json_file(config_file) if config_file is not None else None
    tokenizer=BertTokenizer(vocab_file=vocab_file) if vocab_file is not None else None
    model = BertModel(config=config) if config is not None else None
    return tokenizer,model


def init_xlmr_model(vocab_file, config_file, ckpt_file=None):
    config=XLMRobertaConfig.from_json_file(config_file) if config_file is not None else None
    tokenizer=XLMRobertaTokenizer(vocab_file=vocab_file) if vocab_file is not None else None
    model = XLMRobertaModel(config=config) if config is not None else None
    return tokenizer,model


def init_bertqa_model(vocab_file=None, config_file=None, ckpt_file=None):
    tokenizer=BertTokenizer(vocab_file=vocab_file) if vocab_file is not None else None
    if config_file is not None:
        config=BertConfig.from_json_file(config_file)
        model = BertForQASimple(config=config)
        if ckpt_file is not None:
            sd = torch.load(ckpt_file,map_location='cpu')
            a1, a2 = model.load_state_dict(sd)
            print (a1)
            print (a2)
    else:
        config = None
        model = None
    return tokenizer,model


def extract_sentences_from_cmrc2018(data_file):
    with open(data_file,'r',encoding='utf-8') as f:
        data = json.load(f)
    lines = []
    for example in data:
        lines.append(example['context_text'])
        lines.extend([qa['query_text'] for qa in example['qas']])
    return lines

if __name__ == '__main__':
    data_file = './datasets/cmrc2018-train.json'
    lines = extract_sentences_from_cmrc2018(data_file)
    print("Number of lines: ", len(lines))

    tokenizer, model = init_bertqa_model(chinese_roberta_vocab_file, chinese_roberta_config_file, chinese_roberta_qa_ckpt_file)
    print("Vocab size:",tokenizer.vocab_size)
    print("Embedding size:", model.bert.get_input_embeddings().weight.shape)

    outdir='./pruned-models/'
    pruner = Pruner('bert',model.bert,tokenizer,outdir=outdir)
    _, _ = pruner.prune_embeddings(dataiter=lines,min_count=15)
    torch.save(model.state_dict(),os.path.join(outdir,'PrunedEmbedding.pkl'))

    print("New embedding size:", model.bert.get_input_embeddings().weight.shape)
    new_tokenizer, _ = init_bert_model(outdir+'/vocab.txt')
    print("New vocab size:", new_tokenizer.vocab_size)



