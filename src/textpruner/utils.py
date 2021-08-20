import  torch
import numpy as np
from collections import abc

def move_to_device(batch, device):
    r"""Puts each data field to the device"""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch,(list,tuple)):
        return tuple(move_to_device(item,device) for item in batch)
    elif isinstance(batch, abc.Mapping):
        return {key: move_to_device(value,device) for key, value in batch.items()}
    else:
        return batch


def generate_mask(importance : np.ndarray, target_size : int, is_layerwise : bool=False) -> np.ndarray:

    if is_layerwise is True:
        mask = np.ones_like(importance)
        for i,layer in enumerate(importance):
            importance_layer_order = np.argsort(layer)
            mask[i][importance_layer_order[:-target_size]] = 0
    else: #if is_layerwise_pruning is False:
        importance_flat = importance.reshape(-1)
        importance_order = np.argsort(importance_flat)   # ascending
        mask_flat = np.ones_like(importance_flat)
        for pos in importance_order[:-target_size]:
            mask_flat[pos] = 0
        mask = mask_flat.reshape(importance.shape)
    return mask