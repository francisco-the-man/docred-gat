import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ATLoss(nn.Module):
    """
    Adaptive Thresholding Loss for multi-label classification
    with additional regularization for graph-based learning
    """
    def __init__(self, gnn_regularization=0.01):
        super().__init__()
        self.gnn_regularization = gnn_regularization

    def forward(self, logits, labels, gnn_outputs=None):
        # Original ATLoss computation
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
        
        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)
        
        # Main classification loss
        loss = loss1 + loss2
        loss = loss.mean()

        # Add GAT regularization if provided
        if gnn_outputs is not None and isinstance(gnn_outputs, dict):
            # Structural regularization
            if 'attention_weights' in gnn_outputs:
                # Encourage sparse but meaningful attention
                attn_weights = gnn_outputs['attention_weights']
                sparsity_loss = -torch.mean(torch.sum(
                    attn_weights * torch.log(attn_weights + 1e-10), dim=-1))
                loss = loss + self.gnn_regularization * sparsity_loss
            
            # Optional: Add smoothness regularization
            if 'node_embeddings' in gnn_outputs:
                node_embs = gnn_outputs['node_embeddings']
                if 'edge_index' in gnn_outputs:
                    edge_index = gnn_outputs['edge_index']
                    # Encourage connected nodes to have similar representations
                    src, dst = edge_index
                    smoothness_loss = torch.mean(torch.norm(
                        node_embs[src] - node_embs[dst], p=2, dim=1))
                    loss = loss + self.gnn_regularization * smoothness_loss

        return loss

    def get_label(self, logits, num_labels=-1):
        """Convert logits to predicted labels using adaptive thresholding"""
        th_logit = logits[:, 0].unsqueeze(1)  # threshold is no-relation
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]  # smallest logits among the num_labels
            mask = (logits >= top_v.unsqueeze(1)) & mask
            
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output

    def get_score(self, logits, num_labels=-1):
        """Get prediction scores for evaluation"""
        if num_labels > 0:
            return torch.topk(logits, num_labels, dim=1)
        else:
            return logits[:,1] - logits[:,0], 0
