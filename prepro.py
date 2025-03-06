import json
import pickle
import os
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np

docred_ner2id = json.load(open(os.path.join('DocRED', 'meta', 'ner2id.json'), 'r'))
docred_rel2id = json.load(open(os.path.join('DocRED', 'meta', 'rel2id.json'), 'r'))


############################################
# 1) Replacing Asterisk with "[unused1]"
#    We'll define some constants:
############################################
ENTITY_START_TOKEN = "[unused1]"
ENTITY_END_TOKEN   = "[unused2]"

def add_entity_markers(sample, tokenizer, entity_start, entity_end):
    """
    Insert special tokens around entity boundaries:
      e.g. [unused1] <word> ... <word> [unused1].
    This approach signals to the model that these tokens are part of a named entity.
    """
    sents = []
    sent_map = []
    sent_pos = []

    sent_start = 0
    for i_s, sent in enumerate(sample['sents']):
        new_map = {}
        for i_t, token in enumerate(sent):
            tokens_wordpiece = tokenizer.tokenize(token)

            # If it's the start of an entity
            if (i_s, i_t) in entity_start:
                tokens_wordpiece = [ENTITY_START_TOKEN] + tokens_wordpiece

            # If it's the end of an entity
            if (i_s, i_t) in entity_end:
                tokens_wordpiece = tokens_wordpiece + [ENTITY_END_TOKEN]

            # Record how many wordpieces get appended
            new_map[i_t] = len(sents)
            sents.extend(tokens_wordpiece)

        sent_end = len(sents)
        sent_pos.append((sent_start, sent_end))
        sent_start = sent_end

        # Mark the next token index if needed
        new_map[i_t + 1] = len(sents)
        sent_map.append(new_map)

    return sents, sent_map, sent_pos

def slide_and_chunk(token_ids, max_seq_length, doc_stride=128):
    """
    Sliding window approach with overlap.
    
    :param token_ids: List of token IDs for the entire doc (after adding entity markers but before special tokens).
    :param max_seq_length: The max total tokens for the model input (including special tokens).
    :param doc_stride: How many tokens to shift between windows. 128 is typical if max_seq_length is 512, for instance.
    :return: A list of chunked token lists, each within max_seq_length.
    """
    # We'll reserve 2 tokens for [CLS] + [SEP] (for BERT or RoBERTa)
    chunk_size = max_seq_length - 2
    chunks = []
    start_pos = 0

    while start_pos < len(token_ids):
        end_pos = min(start_pos + chunk_size, len(token_ids))
        chunk = token_ids[start_pos:end_pos]
        chunks.append(chunk)
        if end_pos == len(token_ids):
            break
        # Move the window by doc_stride
        start_pos += doc_stride

    return chunks

def read_docred(file_in,
                tokenizer,
                transformer_type="roberta",
                max_seq_length=512,  # typical for RoBERTa
                doc_stride=128,
                teacher_sig_path="",
                single_results=None):
    """
    Read the DocRED json file, insert entity markers, create features.
    This version uses sliding windows with overlap instead of naive truncation.
    """

    if not file_in or not os.path.exists(file_in):
        print(f"File not found: {file_in}")
        return None

    data = json.load(open(file_in, "r"))
    features = []

    pos_samples = 0
    neg_samples = 0
    i_line = 0

    # teacher signals if needed
    if teacher_sig_path:
        basename = os.path.splitext(os.path.basename(file_in))[0]
        attns_file = os.path.join(teacher_sig_path, f"{basename}.attns")
        attns = pickle.load(open(attns_file, 'rb'))
    else:
        attns = None

    # If we are using single_results for pseudo-labeling
    title2preds = {}
    if single_results is not None:
        for pred_rel in single_results:
            t = pred_rel["title"]
            if t not in title2preds:
                title2preds[t] = []
            title2preds[t].append(pred_rel)

    for doc_id in tqdm(range(len(data)), desc="Loading examples"):
        sample = data[doc_id]

        # The entity list is a list of lists, each sub-list is a cluster of mentions for the same entity
        entities = sample['vertexSet']

        # Build the start/end sets for mention boundaries
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                s_id = mention["sent_id"]
                start_tok = mention["pos"][0]
                end_tok = mention["pos"][1] - 1
                entity_start.append((s_id, start_tok))
                entity_end.append((s_id, end_tok))

        # Insert special tokens [unused1] around entity boundaries
        sents, sent_map, sent_pos = add_entity_markers(sample, tokenizer, entity_start, entity_end)

        # Build gold "train_triple" (DocRED calls them 'labels')
        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                r = docred_rel2id[label['r']]  # mapped to an integer ID
                h, t = label['h'], label['t']
                evi = label['evidence']  # list of sentence indices
                if (h, t) not in train_triple:
                    train_triple[(h, t)] = [{'relation': r, 'evidence': evi}]
                else:
                    train_triple[(h, t)].append({'relation': r, 'evidence': evi})

        # Build entity_pos array: each entity -> list of (start, end) in the final token list
        entity_pos = []
        for e in entities:
            entity_mentions = []
            for m in e:
                s_id = m["sent_id"]
                start_wp = sent_map[s_id][m["pos"][0]]
                end_wp   = sent_map[s_id][m["pos"][1]]
                entity_mentions.append((start_wp, end_wp))
            entity_pos.append(entity_mentions)

        # We'll now produce 'relations' and 'hts'
        # hts = all pairs (h, t)
        # relations = for each pair, a multi-hot vector
        relations = []
        hts = []
        sent_labels = []

        # Positive examples from train_triple
        for (h, t), triple_info_list in train_triple.items():
            relation_vec = [0]*len(docred_rel2id)
            sent_evi = [0]*len(sent_pos)
            for info in triple_info_list:
                relation_vec[info['relation']] = 1
                for s_id in info['evidence']:
                    sent_evi[s_id] += 1
            relations.append(relation_vec)
            hts.append([h, t])
            sent_labels.append(sent_evi)
            pos_samples += 1

        # Negative examples: any pair that doesn't appear in train_triple
        # We'll label them as [NA] => index 0 in docred_rel2id, or do a special encoding
        # Here we assume index 0 is the 'no_relation' or NA label if your rel2id is like:
        # {"NA": 0, "P17":1, "P19":2, ...}
        all_ent = range(len(entities))
        for h in all_ent:
            for t in all_ent:
                if h != t and [h, t] not in hts:
                    relation_vec = [0]*len(docred_rel2id)
                    # typically docred_rel2id['NA'] = 0 => set relation_vec[0]=1
                    relation_vec[0] = 1  
                    sent_evi = [0]*len(sent_pos)
                    relations.append(relation_vec)
                    hts.append([h, t])
                    sent_labels.append(sent_evi)
                    neg_samples += 1

        # Convert tokens -> IDs
        input_ids = tokenizer.convert_tokens_to_ids(sents)

        # Now do sliding window chunking
        # We'll store multiple feature chunks if the doc is > max_seq_length
        # Each chunk gets [CLS] and [SEP] appended by build_inputs_with_special_tokens
        chunked_input = slide_and_chunk(input_ids, max_seq_length, doc_stride=doc_stride)

        # We'll produce a separate feature for each chunk
        doc_features = []
        for chunk_i, chunk_ids in enumerate(chunked_input):
            # Build final input with special tokens
            final_input_ids = tokenizer.build_inputs_with_special_tokens(chunk_ids)

            # We have to map entity mentions to chunk indices
            # We'll do a simple approach: any mention that falls into [start, end) is kept
            chunk_start = chunk_i*doc_stride
            chunk_end   = chunk_start + (max_seq_length - 2)
            # We'll compute entity_pos for this chunk
            chunk_entity_pos = []
            for e_ix, mention_list in enumerate(entity_pos):
                chunk_mentions = []
                for (m_start, m_end) in mention_list:
                    # if mention is wholly in this chunk window
                    if m_start >= chunk_start and m_end <= chunk_end:
                        # shift them relative to chunk_start plus 1 for [CLS]
                        offset = 1  # for roberta/bert, [CLS] is at position 0
                        new_start = m_start - chunk_start + offset
                        new_end   = m_end   - chunk_start + offset
                        # clamp if needed
                        if new_end >= (len(final_input_ids) - 1):
                            continue
                        chunk_mentions.append((new_start, new_end))
                chunk_entity_pos.append(chunk_mentions)

            # Build the final feature for this chunk
            feature_dict = {
                "input_ids": final_input_ids,
                "entity_pos": chunk_entity_pos,
                "labels": relations,       # same set for all chunks
                "hts": hts,
                "sent_pos": sent_pos,      # optional, depends on how you handle multi-chunk
                "sent_labels": sent_labels,
                "title": sample["title"],
                "chunk_idx": chunk_i
            }

            # teacher signals if used
            if attns is not None:
                # e.g. if we have teacher attention => slice it as well
                teacher_attn_slice = attns[doc_id][:, chunk_start:chunk_end]
                feature_dict['attns'] = teacher_attn_slice

            doc_features.append(feature_dict)

        features.extend(doc_features)
        i_line += len(doc_features)

    print("# of documents: {} -> # of feature chunks: {}".format(len(data), i_line))
    print("Positive samples: {} | Negative samples: {}".format(pos_samples, neg_samples))
    return features


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    read_docred("DocRED/train_annotated.json", tokenizer)