import torch
import torch.nn as nn
from opt_einsum import contract
import torch.nn.functional as F
from long_seq import process_long_input
from losses import ATLoss
from torch_geometric.nn import GATConv


class DocREModel(nn.Module):

    def __init__(self, config, model, tokenizer,
                emb_size=768, block_size=64, num_labels=-1,
                max_sent_num=25, evi_thresh=0.2,
                gnn_layers=2, gat_heads=4):
        '''
        Initialize the model.
        :model: Pretrained langage model encoder;
        :tokenizer: Tokenzier corresponding to the pretrained language model encoder;
        :emb_size: Dimension of embeddings for subject/object (head/tail) representations;
        :block_size: Number of blocks for grouped bilinear classification;
        :num_labels: Maximum number of relation labels for each entity pair;
        :max_sent_num: Maximum number of sentences for each document;
        :evi_thresh: Threshold for selecting evidence sentences;
        :gnn_layers: Number of GAT layers;
        :gat_heads: Number of attention heads in each GAT layer.
        '''
        
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = config.hidden_size

        self.loss_fnt = ATLoss()
        self.loss_fnt_evi = nn.KLDivLoss(reduction="batchmean")

        # GAT layers for entity interaction
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size//gat_heads,
                heads=gat_heads,
                concat=True,
                dropout=0.1
            ) for _ in range(gnn_layers)
        ])

        # Entity representation extractors
        self.head_extractor = nn.Linear(self.hidden_size * 2, emb_size)
        self.tail_extractor = nn.Linear(self.hidden_size * 2, emb_size)     

        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.total_labels = config.num_labels
        self.max_sent_num = max_sent_num
        self.evi_thresh = evi_thresh


    def encode(self, input_ids, attention_mask):
        
        '''
        Get the embedding of each token. For long document that has more than 512 tokens, split it into two overlapping chunks.
        Inputs:
            :input_ids: (batch_size, doc_len)
            :attention_mask: (batch_size, doc_len)
        Outputs:
            :sequence_output: (batch_size, doc_len, hidden_dim)
            :attention: (batch_size, num_attn_heads, doc_len, doc_len)
        '''
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        # process long documents.
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts, edge_index, offset):
        '''
        Get head, tail, context embeddings from token embeddings.
        Added edge_index parameter for GAT layers.
        '''
        
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        ht_atts = []

        for i in range(len(entity_pos)): # for each batch
            entity_embs, entity_atts = [], []
            
            # obtain entity embedding from mention embeddings.
            for eid, e in enumerate(entity_pos[i]): # for each entity
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for mid, (start, end) in enumerate(e): # for every mention
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])

                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)

                entity_embs.append(e_emb)
                entity_atts.append(e_att)
                
            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            # Apply GAT layers to entity embeddings
            x = entity_embs
            curr_edge_index = edge_index[i].to(x.device)
            for gat_layer in self.gat_layers:
                x = gat_layer(x, curr_edge_index)
                x = F.elu(x)
            
            # Updated entity embeddings after graph attention
            entity_embs = x

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            
            # obtain subject/object (head/tail) embeddings from updated entity embeddings
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
                
            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30)
            ht_atts.append(ht_att)
            
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        
        rels_per_batch = [len(b) for b in hss]
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        ht_atts = torch.cat(ht_atts, dim=0)

        return hss, rss, tss, ht_atts, rels_per_batch


    def forward_rel(self, hs, ts, rs):
        '''
        Forward computation for RE.
        Inputs:
            :hs: (num_ent_pairs_all_batches, emb_size)
            :ts: (num_ent_pairs_all_batches, emb_size)
            :rs: (num_ent_pairs_all_batches, emb_size)
        Outputs:
            :logits: (num_ent_pairs_all_batches, num_rel_labels)
        '''
        
        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=-1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=-1)))
        # split into several groups.
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)

        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)
        
        return logits


    def forward_evi(self, doc_attn, sent_pos, batch_rel, offset):
        '''
        Forward computation for ER.
        Inputs:
            :doc_attn: (num_ent_pairs_all_batches, doc_len), attention weight of each token for computing localized context pooling.
            :sent_pos: list of list. The outer length = batch size. The inner list contains (start, end) position of each sentence in each batch.
            :batch_rel: list of length = batch size. Each entry represents the number of entity pairs of the batch.
            :offset: 1 for bert and roberta. Offset caused by [CLS] token.
        Outputs:
            :s_attn:  (num_ent_pairs_all_batches, max_sent_all_batch), sentence-level evidence distribution of each entity pair.
        '''
        
        max_sent_num = max([len(sent) for sent in sent_pos])
        rel_sent_attn = []
        for i in range(len(sent_pos)): # for each batch
            # the relation ids corresponds to document in batch i is [sum(batch_rel[:i]), sum(batch_rel[:i+1]))
            curr_attn = doc_attn[sum(batch_rel[:i]):sum(batch_rel[:i+1])]
            curr_sent_pos = [torch.arange(s[0], s[1]).to(curr_attn.device) + offset for s in sent_pos[i]] # + offset

            curr_attn_per_sent = [curr_attn.index_select(-1, sent) for sent in curr_sent_pos]
            curr_attn_per_sent += [torch.zeros_like(curr_attn_per_sent[0])] * (max_sent_num - len(curr_attn_per_sent))
            sum_attn = torch.stack([attn.sum(dim=-1) for attn in curr_attn_per_sent], dim=-1) # sum across those attentions
            rel_sent_attn.append(sum_attn)

        s_attn = torch.cat(rel_sent_attn, dim=0)
        return s_attn


    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None, # relation labels
                entity_pos=None,
                hts=None, # entity pairs
                sent_pos=None, 
                sent_labels=None, # evidence labels (0/1)
                teacher_attns=None, # evidence distribution from teacher model
                edge_index=None,  # Added edge_index parameter
                tag="train",
                ):

        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        output = {}
        sequence_output, attention = self.encode(input_ids, attention_mask)

        hs, rs, ts, doc_attn, batch_rel = self.get_hrt(sequence_output, attention, entity_pos, hts, edge_index, offset)
        logits = self.forward_rel(hs, ts, rs)

        # Collect GAT outputs
        gnn_outputs = {
            'attention_weights': None,  # Store GAT attention weights
            'node_embeddings': None,    # Store node embeddings
            'edge_index': None         # Store edge structure
        }
        
        # During GAT processing, store the outputs
        for gat_layer in self.gat_layers:
            x, attention = gat_layer(sequence_output, edge_index, return_attention_weights=True)
            gnn_outputs['attention_weights'] = attention
            gnn_outputs['node_embeddings'] = x
            gnn_outputs['edge_index'] = edge_index
        
        # Pass GAT outputs to loss function
        loss = self.loss_fnt(logits.float(), labels.float(), gnn_outputs)
        output["loss"] = {"rel_loss": loss.to(sequence_output)}

        if sent_labels != None: # human-annotated evidence available

            s_attn = self.forward_evi(doc_attn, sent_pos, batch_rel, offset)
            output["evi_pred"] = F.pad(s_attn > self.evi_thresh, (0, self.max_sent_num - s_attn.shape[-1]))

        if tag in ["test", "dev"]: # testing
            scores_topk = self.loss_fnt.get_score(logits, self.num_labels)
            output["scores"] = scores_topk[0]
            output["topks"] = scores_topk[1]
        
        if tag == "infer": # teacher model inference
            output["attns"] = doc_attn.split(batch_rel)

        else: # training
            if sent_labels != None: # supervised training with human evidence

                idx_used = torch.nonzero(labels[:,1:].sum(dim=-1)).view(-1)
                # evidence retrieval loss (kldiv loss)
                s_attn = s_attn[idx_used]
                sent_labels = sent_labels[idx_used]
                norm_s_labels = sent_labels/(sent_labels.sum(dim=-1, keepdim=True) + 1e-30)
                norm_s_labels[norm_s_labels == 0] = 1e-30
                s_attn[s_attn == 0] = 1e-30
                evi_loss = self.loss_fnt_evi(s_attn.log(), norm_s_labels)
                output["loss"]["evi_loss"] = evi_loss.to(sequence_output)
            
            elif teacher_attns != None: # self training with teacher attention
                
                doc_attn[doc_attn == 0] = 1e-30
                teacher_attns[teacher_attns == 0] = 1e-30
                attn_loss = self.loss_fnt_evi(doc_attn.log(), teacher_attns)
                output["loss"]["attn_loss"] = attn_loss.to(sequence_output)
        
        return output
