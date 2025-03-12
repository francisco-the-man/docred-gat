import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
import wandb
from tqdm import tqdm
import itertools
import json
import os
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred
from evaluation import to_official, official_evaluate

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Data args
    parser.add_argument("--data_dir", default="./DocRED", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str)
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    
    # GAT tuning args
    parser.add_argument("--gnn_layers_range", default="1,2,3", type=str,
                        help="Comma-separated list of GNN layers to try")
    parser.add_argument("--gat_heads_range", default="2,4,8", type=str,
                        help="Comma-separated list of attention heads to try")
    parser.add_argument("--dropout_range", default="0.1,0.2,0.3", type=str,
                        help="Comma-separated list of dropout values")
    
    # Training args
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    
    return parser.parse_args()

def validate_model(model, dev_dataloader, device):
    """Run validation and return metrics"""
    model.eval()
    total_loss = 0
    preds, evi_preds = [], []
    
    with torch.no_grad():
        for batch in tqdm(dev_dataloader, desc="Validating"):
            # Prepare inputs
            inputs = {
                'input_ids': batch[0].to(device),
                'attention_mask': batch[1].to(device),
                'labels': batch[2],
                'entity_pos': batch[3],
                'hts': batch[4],
                'sent_pos': batch[5],
                'sent_labels': batch[6].to(device) if batch[6] is not None else None,
                'edge_index': batch[8],
                'tag': 'dev'
            }
            
            outputs = model(**inputs)
            pred = outputs["rel_pred"]
            pred = pred.cpu().numpy()
            preds.append(pred)
            
            if "evi_pred" in outputs:
                evi_pred = outputs["evi_pred"].cpu().numpy()
                evi_preds.append(evi_pred)
    
    preds = np.concatenate(preds, axis=0)
    if evi_preds:
        evi_preds = np.concatenate(evi_preds, axis=0)
    
    official_results = to_official(preds, dev_features, evi_preds=evi_preds)
    best_re, best_evi, best_re_ign, _ = official_evaluate(
        official_results, 
        args.data_dir, 
        "train.json", 
        args.dev_file
    )
    
    metrics = {
        "f1": best_re[-1],
        "f1_ign": best_re_ign[-1],
        "evi_f1": best_evi[-1] if best_evi else 0
    }
    
    return metrics

def main():
    args = parse_args()
    wandb.init(project="DocRED-GAT-Tuning")
    
    # Set up device and seed
    device = torch.device(args.device)
    set_seed(args)
    
    # Load model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    bert_model = AutoModel.from_pretrained(args.model_name_or_path)
    
    # Prepare validation data
    dev_file = os.path.join(args.data_dir, args.dev_file)
    dev_features = read_docred(dev_file, tokenizer, max_seq_length=args.max_seq_length)
    dev_dataloader = DataLoader(
        dev_features, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        drop_last=False
    )
    
    # Parse parameter ranges
    gnn_layers_range = [int(x) for x in args.gnn_layers_range.split(",")]
    gat_heads_range = [int(x) for x in args.gat_heads_range.split(",")]
    dropout_range = [float(x) for x in args.dropout_range.split(",")]
    
    # Generate all combinations
    param_combinations = list(itertools.product(
        gnn_layers_range,
        gat_heads_range,
        dropout_range
    ))
    
    # Results storage
    results = []
    best_f1 = 0
    best_params = None
    
    # Try each combination
    for gnn_layers, gat_heads, dropout in tqdm(param_combinations, desc="Testing combinations"):
        config_dict = {
            "gnn_layers": gnn_layers,
            "gat_heads": gat_heads,
            "dropout": dropout
        }
        print(f"\nTesting configuration: {config_dict}")
        
        # Initialize model with current parameters
        model = DocREModel(
            config=config,
            model=bert_model,
            tokenizer=tokenizer,
            gnn_layers=gnn_layers,
            gat_heads=gat_heads,
            dropout_rate=dropout
        )
        model.to(device)
        
        # Validate
        metrics = validate_model(model, dev_dataloader, device)
        
        # Log results
        result = {
            **config_dict,
            **metrics
        }
        results.append(result)
        wandb.log(result)
        
        print(f"F1: {metrics['f1']:.4f}, F1_ign: {metrics['f1_ign']:.4f}, Evi_F1: {metrics['evi_f1']:.4f}")
        
        # Track best configuration
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_params = config_dict
    
    # Save results
    output = {
        "all_results": results,
        "best_params": best_params,
        "best_f1": best_f1
    }
    
    with open("gat_tuning_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\nBest configuration:")
    print(json.dumps(best_params, indent=2))
    print(f"Best F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()
