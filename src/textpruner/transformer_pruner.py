

import torch
from torch import nn
import numpy as np
import os
from .configurations import MODEL_MAP, TransformerPruningConfig, EmbeddingPruningConfig, GeneralConfig
from .utils import move_to_device, generate_mask
import logging
from tqdm import tqdm
from collections import abc
from typing import Optional

logger = logging.getLogger(__name__)

class TransformerPruner:
    def __init__(self, model : nn.Module, 
                       general_config : GeneralConfig,
                       transformer_pruning_config : TransformerPruningConfig,
                       base_model_prefix : Optional[str] = None):

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
                raise ValueError("Cannot infer/get model_type! Maybe you should provide base_model_prefix") 

        self.general_config = general_config
        self.transformer_pruning_config = transformer_pruning_config

        self.output_dir = self.general_config.output_dir
    
        # They are none unless after pruning
        self.head_mask = None
        self.ffn_mask = None

        os.makedirs(self.output_dir, exist_ok=True)

    def prune_transformer(self, dataloader, adaptor, batch_postprocessor, keep_weights=False):
        if self.transformer_pruning_config.is_iterative is False:
            self.prune_transformer_once(dataloader, adaptor, batch_postprocessor, keep_weights=keep_weights)
        else:
            self.prune_transformer_iterative(dataloader, adaptor, batch_postprocessor, keep_weights=keep_weights)

    def prune_transformer_with_masks(self,head_mask=None, ffn_mask=None, keep_shape=False):
        if head_mask is None:
            head_mask = self.head_mask
        if ffn_mask is None:
            ffn_mask = self.ffn_mask

        if ffn_mask is not None:
            ffn_mask_tensor = torch.tensor(ffn_mask).to(dtype=torch.float32, device=self.general_config.device)
            reorder_ffn_weights(self.base_model, ffn_mask_tensor, keep_shape)
        if head_mask is not None:
            if keep_shape:
                head_mask_tensor = torch.tensor(head_mask).to(dtype=torch.float32, device=self.general_config.device)
                reorder_attention_heads(self.base_model, head_mask_tensor, keep_shape)
            heads_to_prune_dict = {}
            for layer_num, layer_head in enumerate(head_mask.tolist()):
                heads_to_prune_dict[layer_num] = []
                for head_idx, v in enumerate(layer_head):
                    if v==0:
                        heads_to_prune_dict[layer_num].append(head_idx)
            print(heads_to_prune_dict)
            if not keep_shape:
                self.base_model.prune_heads(heads_to_prune_dict)

    def prune_transformer_iterative(self, dataloader, adaptor, batch_postprocessor, keep_weights=False):

        ffn_size = self.transformer_pruning_config.ffn_size
        num_of_heads = self.transformer_pruning_config.num_of_heads
        n_iter = self.transformer_pruning_config.n_iter

        head_importance_fn = os.path.join(self.output_dir, f'head_importance.npy')
        ffn_importance_fn = os.path.join(self.output_dir,f'ffn_importance.npy')

        if os.path.exists(head_importance_fn) and os.path.exists(ffn_importance_fn):
            logger.info(f"Loading pre-cached head importance score {head_importance_fn}")
            head_importance = np.load(head_importance_fn)
            logger.info(f"Loading pre-cached ffn importance score {ffn_importance_fn}")
            ffn_importance = np.load(ffn_importance_fn)
        else:
            logger.info("Calculating head importance and ffn importance")
            head_importance, ffn_importance = self.get_ffn_and_head_importance_score(dataloader, adaptor, batch_postprocessor)
            head_importance = head_importance.cpu().numpy() # (num_layers, num_heads)
            ffn_importance = ffn_importance.cpu().numpy() # (num_layers, intermediate_size)
            # Save importance score
            logger.info("save...")
            np.save(head_importance_fn, head_importance)
            np.save(ffn_importance_fn, ffn_importance)

        ori_num_of_heads = head_importance.shape[0]*head_importance.shape[1]
        ori_ffn_size = ffn_importance.shape[0]*ffn_importance.shape[1]

        ffn_size_per_iter = (ori_ffn_size - ffn_size)//n_iter
        num_of_heads_per_iter = (ori_num_of_heads - num_of_heads)//n_iter
        ffn_size_res = (ori_ffn_size - ffn_size)%n_iter
        num_of_heads_res = (ori_num_of_heads - num_of_heads)%n_iter

        dffn_size = ori_ffn_size
        dnum_of_heads = ori_num_of_heads

        for i in range(n_iter):

            print('n_iter:', i)

            if i > 0:
                logger.info("Calculating head importance and ffn importance")
                head_importance, ffn_importance = self.get_ffn_and_head_importance_score(dataloader, adaptor, batch_postprocessor)
                head_importance = head_importance.cpu().numpy() # (num_layers, num_heads)
                ffn_importance = ffn_importance.cpu().numpy() # (num_layers, intermediate_size)

            dffn_size -= ffn_size_per_iter + 1 if i < ffn_size_res else ffn_size_per_iter
            dnum_of_heads -= num_of_heads_per_iter + 1 if i < num_of_heads_res else num_of_heads_per_iter

            self.head_mask = generate_mask(head_importance, dnum_of_heads, self.transformer_pruning_config.head_is_layerwise)
            self.ffn_mask = generate_mask(ffn_importance, dffn_size, self.transformer_pruning_config.ffn_is_layerwise)

            logger.info(f"New ffn size:{self.ffn_mask.sum(-1).tolist()}")
            logger.info(f"New num heads:{self.head_mask.sum(-1).tolist()}")

            logger.info("Head and ffn masks have been generated, can be accessed via self.head_mask and self.ffn_mask")
            if keep_weights is False:
                logger.info("Remove redundant weights from the model")
                if i == n_iter-1:
                    self.prune_transformer_with_masks()
                else:
                    self.prune_transformer_with_masks(keep_shape=True)

    def prune_transformer_once(self,dataloader, adaptor, batch_postprocessor, keep_weights=False):

        head_importance_fn = os.path.join(self.output_dir, f'head_importance.npy')
        ffn_importance_fn = os.path.join(self.output_dir,f'ffn_importance.npy')

        if os.path.exists(head_importance_fn) and os.path.exists(ffn_importance_fn):
            logger.info(f"Loading pre-cached head importance score {head_importance_fn}")
            head_importance = np.load(head_importance_fn)
            logger.info(f"Loading pre-cached ffn importance score {ffn_importance_fn}")
            ffn_importance = np.load(ffn_importance_fn)
        else:
            logger.info("Calculating head importance and ffn importance")
            head_importance,ffn_importance = self.get_ffn_and_head_importance_score(dataloader, adaptor, batch_postprocessor)
            head_importance = head_importance.cpu().numpy() # (num_layers, num_heads)
            ffn_importance = ffn_importance.cpu().numpy() # (num_layers, intermediate_size)
            # Save importance score
            logger.info("save...")
            np.save(head_importance_fn, head_importance)
            np.save(ffn_importance_fn, ffn_importance)


        ffn_size = self.transformer_pruning_config.ffn_size
        num_of_heads = self.transformer_pruning_config.num_of_heads

        self.head_mask = generate_mask(head_importance, num_of_heads, self.transformer_pruning_config.head_is_layerwise)
        self.ffn_mask = generate_mask(ffn_importance, ffn_size, self.transformer_pruning_config.ffn_is_layerwise)


        logger.info(f"New ffn size:{self.ffn_mask.sum(-1).tolist()}")
        logger.info(f"New num heads:{self.head_mask.sum(-1).tolist()}")

        logger.info("Head and ffn masks have been generated, can be accessed via self.head_mask and self.ffn_mask")
        if keep_weights is False:
            logger.info("Remove redundant weights from the model")
            self.prune_transformer_with_masks()

    def save_masks(self):
        output_dir = os.path.join(self.general_config.output_dir,f'pruned_Head_FFN_masks')
        os.makedirs(output_dir, exist_ok=True)

        if torch.__version__ >= '1.6':
            torch.save((self.head_mask,self.ffn_mask),os.path.join(output_dir,f'masks.pkl'),_use_new_zipfile_serialization=False)
        else:
            torch.save((self.head_mask,self.ffn_mask),os.path.join(output_dir,f'model.pkl'))
        # save config
        logger.info(f"Masks have been saved to {output_dir}")

    def save_model(self):
        if self.transformer_pruning_config.ffn_is_layerwise is False:
            raise NotImplementedError("Can not save pruned model with ffn_is_layerwise False. Use save_masks")
        #save pruned model
        ffn_size = self.ffn_mask.astype(int).sum(-1).tolist()[0]
        self.base_model.config.intermediate_size = ffn_size

        num_of_heads = self.head_mask.astype(int).sum().item()
        #num_of_heads = self.head_mask.sum(-1).tolist()[0]
        #self.base_model.config.num_attention_heads = num_of_heads

        #config_filename = os.path.join(args.output_dir, f'pruned_{new_num_heads}H')
        #base_model.config.save_pretrained(config_filename)
        #torch.save(model.state_dict(), ckpt_filename)
        #self.model.save_pretrained(ckpt_filename)
        output_dir = os.path.join(self.general_config.output_dir,f'pruned_{num_of_heads}H{ffn_size}FFN')
        os.makedirs(output_dir, exist_ok=True)

        if torch.__version__ >= '1.6':
            torch.save(self.model.state_dict(),os.path.join(output_dir,f'model.pkl'),_use_new_zipfile_serialization=False)
        else:
            torch.save(self.model.state_dict(),os.path.join(output_dir,f'model.pkl'))
        # save config
        self.base_model.config.save_pretrained(output_dir)
        logger.info(f"Model and configuration have been saved to {output_dir}")


    def get_ffn_and_head_importance_score(self, dataloader,
                                adaptor=None, batch_postprocessor=None,
                                head_mask : torch.Tensor = None) -> torch.Tensor :
        model = self.model
        n_layers, n_heads = self.base_model.config.num_hidden_layers, self.base_model.config.num_attention_heads
        intermediate_size = self.base_model.config.intermediate_size
        hidden_size = self.base_model.config.hidden_size

        device = self.general_config.device

        logger.info("***** Running Forward and Backward from get_ffn_and_head_importance_score*****")
        logger.info(" Length of dataloader = %d", len(dataloader))
        model.eval()

        head_importance = torch.zeros(n_layers, n_heads).to(device)
        if head_mask is None:
            head_mask = torch.ones(n_layers, n_heads).to(device)
        else:
            head_mask = head_mask.to(device)
        head_mask.requires_grad_(True)

        #get ffn weights and bias
        inter_weights = [] # torch.zeros(n_layers, intermediate_size, hidden_size).to(device)
        inter_biases = [] #torch.zeros(n_layers, intermediate_size).to(device)
        output_weights = [] #torch.zeros(n_layers, hidden_size, intermediate_size).to(device)
        layers = self.base_model.encoder.layer
        for layer_num in range(n_layers):
                inter_weights.append(layers[layer_num].intermediate.dense.weight) #.detach().to(device)
                inter_biases.append(layers[layer_num].intermediate.dense.bias) #.detach().to(device)
                output_weights.append(layers[layer_num].output.dense.weight) #.detach().to(device)
        ffn_importance = torch.zeros(n_layers, intermediate_size).to(device) #ex. (12,3072) importance for each intermediate dim
        num_examples = 0.0

        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch_postprocessor is not None:
                batch = batch_postprocessor(batch)
            batch = move_to_device(batch, device)
            outputs = model(**batch, head_mask=head_mask)

            if adaptor is None:
                try:
                    if isinstance(batch, abc.Mapping):
                        tmp_eval_loss = outputs['loss']
                    else:
                        tmp_eval_loss = outputs.loss
                except (KeyError, AttributeError) as e:
                    logger.error("Cannot get loss from the output automatically! Adaptor is needed")
                    raise e
            else:
                results = adaptor(outputs, batch)
                tmp_eval_loss = results['loss']
            #tmp_eval_loss, logits = outputs[:2]

            tmp_eval_loss.backward()

            head_importance += head_mask.grad.abs().detach() #abs


            for layer_num in range(n_layers):
                weight1 = inter_weights[layer_num]
                bias1 = inter_biases[layer_num]
                weight2 = output_weights[layer_num]
                ffn_importance[layer_num] += ((weight1.grad * weight1).sum(dim=1)+ bias1.grad * bias1).abs().detach()
                ffn_importance[layer_num] += ((weight2.grad * weight2).sum(dim=0)).abs().detach()


            head_mask.grad.zero_()
            model.zero_grad()
            num_examples += len(batch["attention_mask"])

        head_importance.abs_()
        head_importance /= num_examples
        ffn_importance /= num_examples

        return head_importance * head_mask.detach(), ffn_importance




def reorder_attention_heads(base_model, head_mask, keep_shape = False):

    num_heads = base_model.config.num_attention_heads
    num_layers = head_mask.size(0)
    head_size = int(base_model.config.hidden_size / base_model.config.num_attention_heads)
    

    print(f"Num_heads:{num_heads}")
    print(f"Num_layers:{num_layers}")
    print(f"Head_size:{head_size}")

    #assert torch.all(new_num_heads_vec==new_num_heads_vec[0]), "Numbers of heads in each layer must be equal"

    layers = base_model.encoder.layer
    for layer_num in range(num_layers):
        query_weight = layers[layer_num].attention.self.query.weight
        query_bias = layers[layer_num].attention.self.query.bias
        key_weight = layers[layer_num].attention.self.key.weight
        key_bias = layers[layer_num].attention.self.key.bias
        value_weight = layers[layer_num].attention.self.value.weight
        value_bias = layers[layer_num].attention.self.value.bias
        output_weight = layers[layer_num].attention.output.dense.weight

        # sort query, key, value based on the confidence scores
        query_weight, query_bias = rearange_weights(query_weight,query_bias,head_mask[layer_num],head_size,keep_shape)
        layers[layer_num].attention.self.query.weight = torch.nn.Parameter(query_weight)
        layers[layer_num].attention.self.query.bias = torch.nn.Parameter(query_bias)
        key_weight, key_bias = rearange_weights(key_weight,key_bias,head_mask[layer_num],head_size,keep_shape)
        layers[layer_num].attention.self.key.weight = torch.nn.Parameter(key_weight)
        layers[layer_num].attention.self.key.bias = torch.nn.Parameter(key_bias)
        value_weight, value_bias = rearange_weights(value_weight,value_bias,head_mask[layer_num],head_size,keep_shape)
        layers[layer_num].attention.self.value.weight = torch.nn.Parameter(value_weight)
        layers[layer_num].attention.self.value.bias = torch.nn.Parameter(value_bias)

        output_weight, _ = rearange_weights(output_weight.transpose(0,1), None, head_mask[layer_num],head_size,keep_shape)
        output_weight = output_weight.transpose(0,1)
        layers[layer_num].attention.output.dense.weight = torch.nn.Parameter(output_weight)

def reorder_ffn_weights(base_model, ffn_mask, keep_shape = False):

    num_layers = ffn_mask.size(0)
    head_size = 1 #int(base_model.config.hidden_size / base_model.config.num_attention_heads)

    #print(f"Num_heads:{num_heads}")
    print(f"Num_layers:{num_layers}")

    #assert torch.all(new_ffn_size_vec==new_ffn_size_vec[0]), "Numbers of ffn_size in each layer must be equal"

    layers = base_model.encoder.layer
    for layer_num in range(num_layers):

        inter_weight =layers[layer_num].intermediate.dense.weight
        inter_bias = layers[layer_num].intermediate.dense.bias
        output_weight = layers[layer_num].output.dense.weight

        # sort query, key, value based on the confidence scores
        inter_weight, inter_bias = rearange_weights(inter_weight, inter_bias, ffn_mask[layer_num], head_size, keep_shape)
        layers[layer_num].intermediate.dense.weight = torch.nn.Parameter(inter_weight)
        layers[layer_num].intermediate.dense.bias = torch.nn.Parameter(inter_bias)

        output_weight, _ = rearange_weights(output_weight.transpose(0,1), None, ffn_mask[layer_num], head_size, keep_shape)
        output_weight = output_weight.transpose(0,1)
        layers[layer_num].output.dense.weight = torch.nn.Parameter(output_weight)

def rearange_weights(weight, bias, mask, head_size, keep_shape = False):
    num_heads = mask.size(0)
    mask_dim3 = mask.view(num_heads,1,1).to(torch.bool) # 12,1,1 ?
    weight_dim3 = weight.view(num_heads,head_size,weight.size(1)) # 12,64,768
    if keep_shape == False:
        selected_weight = weight_dim3.masked_select(mask_dim3)
        new_num_heads = int(mask.sum().item())
    else:
        selected_weight = torch.mul(weight_dim3, mask_dim3)
        new_num_heads = num_heads

    ##reshape back
    selected_weight = selected_weight.view(new_num_heads*head_size, weight.size(1))

    selected_bias = None
    if bias is not None:
        mask_dim2 = mask.view(num_heads,1).to(torch.bool) # 12,1 ?
        bias_dim2 = bias.view(num_heads,head_size) #12,64
        if keep_shape == False:
            selected_bias = bias_dim2.masked_select(mask_dim2)
        else:
            selected_bias = torch.mul(bias_dim2, mask_dim2)
        selected_bias = selected_bias.view(new_num_heads*head_size)

    return selected_weight, selected_bias

