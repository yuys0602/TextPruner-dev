import pickle
import os
import logging
logger = logging.getLogger(__name__)
import torch
from torch.utils.data import TensorDataset, Dataset, ConcatDataset
from typing import List
import csv, json
import random
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score


class MultilingualNLIDataset(Dataset):
    def __init__(self, task: str, data_dir: str, split: str, prefix: str, max_seq_length: int, langs: List[str], local_rank: int, tokenizer=None):
        print("Init NLIDataset")
        self.split = split
        self.processor = processors[task]()
        self.output_mode = output_modes[task]
        self.cached_features_files = {lang : os.path.join(data_dir, f'{prefix}_{split}_{max_seq_length}_{lang}.tensor') for lang in langs}
        self.lang_datasets = {}
        lang_features_tensor = {}


        for lang, cached_features_file in self.cached_features_files.items():
            if os.path.exists(cached_features_file):
                logger.info("Loading features from cached file %s", cached_features_file)
                features_tensor = torch.load(cached_features_file)
            else:
                logger.info("Creating features from dataset file at %s", cached_features_file)
                label_list = self.processor.get_labels()
                if split=='train':
                    examples = self.processor.get_train_examples(lang, data_dir)
                elif split=='dev':
                    examples = self.processor.get_dev_examples(lang, data_dir)
                elif split=='test':
                    examples = self.processor.get_test_examples(lang, data_dir)
                else:
                    raise ValueError
                features_tensor = trans3_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, self.output_mode)
                if local_rank in [-1, 0]:
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(features_tensor, cached_features_file)
            features_tensor = features_tensor[:-1] + (features_tensor[-1].long(),)
            self.lang_datasets[lang] = TensorDataset(*features_tensor)
        self.all_dataset = ConcatDataset(list(self.lang_datasets.values()))

    def __getitem__(self, index):
        return self.all_dataset[index]

    def __len__(self):
        return len(self.all_dataset)



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class XnliProcessor(DataProcessor):

    def get_dev_examples(self, lang, data_dir):
        examples = []
        input_file = os.path.join(data_dir,'xnli.dev.jsonl')
        with open(input_file,'r',encoding='utf-8-sig') as f:
           for index,line in enumerate(f):
               raw_example = json.loads(line)
               if raw_example['language'] != lang:
                   continue
               else:
                   text_a = raw_example['sentence1']
                   text_b = raw_example['sentence2']
                   label  = raw_example['gold_label']
                   guid   = f"dev-{index}"
                   examples.append(
                       InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def get_train_examples(self, lang, data_dir):
        examples = []
        input_file = os.path.join(data_dir,f'multinli.train.{lang}.tsv')
        with open(input_file,'r',encoding='utf-8-sig') as f:
            for index,line in enumerate(f):
                if index == 0:
                    continue
                line = line.strip().split('\t')
                guid = f"train-{index}"
                text_a = line[0]
                text_b = line[1]
                label = line[2]
                if label=='contradictory':
                    label = 'contradiction'
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, lang, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "xnli.test.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            language = line[0]
            if language != lang:
                continue
            guid = "%s-%s" % ("test", i)
            text_a = line[6]
            text_b = line[7]
            label = line[1]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class PawsxProcessor(DataProcessor):

    def get_dev_examples(self, lang, data_dir):
        examples = []
        input_file = os.path.join(data_dir,f'dev-{lang}.tsv')
        with open(input_file,'r',encoding='utf-8-sig') as f:
           for index,line in enumerate(f):  
                text_a,text_b, label = line.strip().split('\t')
                guid   = f"dev-{index}"
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self,lang,data_dir):
        examples = []
        input_file = os.path.join(data_dir,f'test-{lang}.tsv')
        with open(input_file,'r',encoding='utf-8-sig') as f:
           for index,line in enumerate(f):  
                text_a,text_b, label = line.strip().split('\t')
                guid   = f"test-{index}"
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ["0","1"]

    def get_train_examples(self, lang, data_dir):
        examples = []
        input_file = os.path.join(data_dir,f'translate-train/{lang}.tsv')
        with open(input_file,'r',encoding='utf-8-sig') as f:
            for index,line in enumerate(f):
                line = line.strip().split('\t')
                if len(line)==5:
                    text_a = line[2]
                    text_b = line[3]
                    label = line[4]
                elif len(line)==3:
                    text_a = line[0]
                    text_b = line[1]
                    label = line[2]
                else:
                    raise ValueError
                guid = f"train-{index}"
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


from dataclasses import dataclass
import dataclasses
from typing import List,Optional,Union
@dataclass(frozen=True)
class Trans3InputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        return json.dumps(dataclasses.asdict(self)) + "\n"


def trans3_convert_examples_to_features(examples, label_list, max_length,
                                 tokenizer, output_mode):
    if max_length is None:
        max_length = tokenizer.max_len

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample):
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_token_type_ids=True
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = Trans3InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    logger.info(f"Convert featrues to tensors")
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    #if self.output_mode == "classification":
    #    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    #elif self.output_mode == "regression":
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    features_tensor = (all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)
    return features_tensor


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "xnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "lcqmc":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

processors = {
    "xnli": XnliProcessor,
    "pawsx": PawsxProcessor
}

output_modes = {
    "xnli": "classification",
    "lcqmc":"classification",
    "pawsx":"classification",
    "amazon":"classification",
}