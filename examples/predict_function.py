import numpy as np
import os
import torch
from torch.utils.data import SequentialSampler,DistributedSampler,DataLoader
from utils import compute_metrics
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)



def predict(model,eval_datasets, step, eval_langs, output_dir, device, predict_batch_size=8, head_mask = None, in_lang = None):
    eval_task = 'xnli'
    eval_output_dir = output_dir
    lang_results = {}
    for lang,eval_dataset in zip(eval_langs, eval_datasets):
        logger.info("Predicting...")
        logger.info("***** Running predictions *****")
        logger.info(" task name = %s", eval_task)
        logger.info(" lang : %s", lang)
        logger.info("  Num  examples = %d", len(eval_dataset))
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=predict_batch_size)
        model.eval()

        pred_logits = []
        label_ids = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids, attention_mask, token_type_ids, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            inputs={'input_ids':input_ids, 'attention_mask':attention_mask,
                    'token_type_ids':token_type_ids}
            with torch.no_grad():
                outputs = model(**inputs, head_mask = head_mask)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
            pred_logits.append(logits.detach().cpu())
            label_ids.append(labels)
        pred_logits = np.array(torch.cat(pred_logits),dtype=np.float32)
        label_ids = np.array(torch.cat(label_ids),dtype=np.int64)

        preds = np.argmax(pred_logits, axis=1)
        results = compute_metrics(eval_task, preds, label_ids)

        logger.info("***** Eval results {} Lang {} *****".format(step, lang))
        for key in sorted(results.keys()):
            logger.info(f"{lang} {key} = {results[key]:.5f}")
        lang_results[lang] = results

    #output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    #write_results(output_eval_file,step,lang_results, eval_langs, in_lang=in_lang)
    model.train()
    return lang_results



####### THE FOLLOWING HAS NOT BEEN ADAPTED#########

def write_results(output_eval_file,step,lang_results, eval_langs, in_lang):
    if in_lang is None:
        in_lang = []
    with open(output_eval_file, "a") as writer:
            writer.write(f"step: {step:<8d} ")
            line = "Acc:"

            in_acc = zs_acc = all_acc = 0
            in_count = zs_count = all_count =0

            #avg_acc = 0
            for lang in eval_langs:
                acc = lang_results[lang]['acc']
                #avg_acc += acc
                all_acc += acc
                all_count += 1
                if lang in in_lang:
                    in_acc += acc
                    in_count += 1
                else:
                    zs_acc += acc
                    zs_count += 1
                line += f"{lang}={acc:.5f} "
            #avg_acc /= len(eval_langs)
            all_acc /= all_count
            if in_count > 0:
                in_acc /= in_count
            if zs_count > 0:
                zs_acc /= zs_count
            line += f"IN/ZS/All={in_acc:.5f} {zs_acc:.5f} {all_acc:.5f}\n"
            writer.write(line)