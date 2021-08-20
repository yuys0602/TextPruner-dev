
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import XLMRobertaTokenizer, XLMRobertaConfig, XLMRobertaModel
from textpruner import Pruner

chinese_roberta_vocab_file='./pretrained-models/chinese-roberta-base/vocab.txt'
chinese_roberta_config_file='./pretrained-models/chinese-roberta-base/bert_config.json'

bert_vocab_file='./pretrained-models/bert-base-uncased/vocab.txt'
bert_config_file='./pretrained-models/bert-base-uncased/config.json'

xlmr_vocab_file='./pretrained-models/xlm-r-base/sentencepiece.bpe.model'
xlmr_config_file='./pretrained-models/xlm-r-base/config.small.json'

def init_bert_model(vocab_file, config_file):
    config=BertConfig.from_json_file(config_file)
    tokenizer=BertTokenizer(vocab_file=vocab_file)
    model = BertModel(config=config)
    return tokenizer,model


def init_xlmr_model(vocab_file, config_file):
    config=XLMRobertaConfig.from_json_file(config_file)
    tokenizer=XLMRobertaTokenizer(vocab_file=vocab_file)
    model = XLMRobertaModel(config=config)
    return tokenizer,model

def init_roberta_model():
    pass


if __name__ == '__main__':
    tokenizer, model = init_xlmr_model(xlmr_vocab_file, xlmr_config_file)
    #tokenizer, model = init_bert_model(chinese_roberta_vocab_file, chinese_roberta_config_file)
    pruner = Pruner('xlmr',model,tokenizer)
    dataiter= ['今日微软发布了一种新的Python语言服务器，称为Pylance，',
         '它是使用语言服务器协议与Visual Studio Code（VS Code）进行相互通信。',
         '微软同时还表示，',
         '新的Pylance语言服务器将使VS Code的Python开发人员的工作效率大大提高.',
         ]


    print("Before pruning:")
    for i in dataiter:
        print (i)
        tokens = tokenizer.tokenize(i)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        print (ids)

    token_ids, model = pruner.prune_embeddings(dataiter=dataiter,min_count=3)
    print (tokenizer.convert_ids_to_tokens(token_ids))
    print (model.get_input_embeddings().weight.shape)

    tokenizer=XLMRobertaTokenizer(vocab_file='./pruned-models/sentencepiece.bpe.model')
    #tokenizer=BertTokenizer(vocab_file='./pruned-models/vocab.txt')
    print ("After pruning:")
    for i in dataiter:
        print (i)
        tokens = tokenizer.tokenize(i)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        print (ids)
