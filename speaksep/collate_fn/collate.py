import logging
from typing import List, Iterable
import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def pad_merge_tensors(tensors: List[torch.Tensor], mx_length: int, fill_with=0):
    tensors = [F.pad(el, (0, mx_length - el.shape[-1]), value=fill_with) for el in tensors]
    return torch.cat(tensors)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {}
    special_keys = {'duration', 'audio_path', 'speaker_id'}
    all_keys = set(dataset_items[0].keys())
    for key in special_keys:
        result_batch[key] = [el[key] for el in dataset_items]
    result_batch['speaker_id'] = torch.tensor(result_batch['speaker_id'])

    for key in all_keys.difference(special_keys):
        tensors = [el[key] for el in dataset_items]
        lengths = [el.shape[-1] for el in tensors]
        
        result_batch[key + '_length'] = torch.tensor(lengths)
        value = 0 if 'spectrogram' not in key else -20
        result_batch[key] = pad_merge_tensors(tensors, mx_length=max(lengths), fill_with=value)
    return result_batch