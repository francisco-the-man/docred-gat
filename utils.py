import torch
import os
import random
import numpy as np



def create_directory(d):
    if d and not os.path.exists(d):
        os.makedirs(d)
    return d

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    """
    Collate function for DataLoader that handles graph structures.
    """
    max_len = max([len(f["input_ids"]) for f in batch])
    
    # Basic inputs
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    
    # Entity and relation info
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    sent_pos = [f["sent_pos"] for f in batch]
    
    # Graph structure
    edge_index = [f["edge_index"] for f in batch]
    
    # Optional evidence supervision
    sent_labels = [f["sent_labels"] for f in batch if "sent_labels" in f]
    if not sent_labels:
        sent_labels = None
        
    # Optional teacher attention signals
    attns = [f["attns"] for f in batch if "attns" in f]
    if not attns:
        attns = None
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    
    if sent_labels and None not in sent_labels:
        max_sent = max([len(f["sent_pos"]) for f in batch])
        sent_labels_tensor = []
        for sent_label in sent_labels:
            sent_label = np.array(sent_label)
            sent_labels_tensor.append(np.pad(sent_label, ((0, 0), (0, max_sent - sent_label.shape[1]))))
        sent_labels_tensor = torch.from_numpy(np.concatenate(sent_labels_tensor, axis=0))
    else:
        sent_labels_tensor = None
    
    if attns:
        attns = [np.pad(attn, ((0, 0), (0, max_len - attn.shape[1]))) for attn in attns]
        attns = torch.from_numpy(np.concatenate(attns, axis=0))
    else:
        attns = None

    return [input_ids, input_mask, labels, entity_pos, hts, sent_pos, sent_labels_tensor, attns, edge_index]