# coding: utf-8
"""
KnowledgeGraph
"""
import os
import brain.config as config
import pkuseg
import numpy as np
from uer.utils.tokenizer import * 

str2tokenizer = {"char": CharTokenizer, "space": SpaceTokenizer, "bert": BertTokenizer}

class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    use pkuseg tokenizer for chinese corpus
    use AutoTokenizer (from huggingface) for english words (or NLTK)
    """

    def __init__(self, spo_files, tokenizer='', predicate=False):
        self.Using_pkuseg= False
        if not tokenizer:
            self.Using_pkuseg = True
            self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=self.segment_vocab)
        else:
            self.tokenizer = tokenizer
        self.predicate = predicate
        self.spo_file_paths = [config.KGS.get(f, f) for f in spo_files]
        self.lookup_table = self._create_lookup_table()
        self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG

        self.special_tags = set(config.NEVER_SPLIT_TAG)

    def _create_lookup_table(self):
        lookup_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        subj, pred, obje = line.strip().split("\t")    # medKG which written in english, need to be adopted here
                        subj, pred, obje = subj.replace('_',' '), pred.replace('_',' '), obje.replace('_', '')
                        if not self.Using_pkuseg:
                            subj = ' '.join(self.tokenizer.tokenize(subj))
                            pred = ' '.join(self.tokenizer.tokenize(pred))
                            obje = ' '.join(self.tokenizer.tokenize(obje))
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                    if self.predicate:
                        value = pred +' '+ obje
                    else:
                        value = obje
                    if subj in lookup_table.keys():
                        lookup_table[subj].add(value)
                    else:
                        lookup_table[subj] = set([value])
        return lookup_table

    def add_knowledge_with_vm(self, sent_batch, max_entities=config.MAX_ENTITIES, add_pad=True, max_length=128):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """
        if self.Using_pkuseg:
            split_sent_batch = [self.tokenizer.cut(sent) for sent in sent_batch]
        else:
            split_sent_batch = [self.tokenizer.tokenize(sent) for sent in sent_batch]
        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
        for split_sent in split_sent_batch:

            # create tree
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []

            all_tokens = []
            all_entities = []
            ''' 
            for token in split_sent:
                keys = [x if token in x else '' for x in self.lookup_table.keys()]
                keys = [x for x in keys if x != '']
                if len(keys) >1:
                    key = keys[0]
                else: key = token
                entities = list(self.lookup_table.get(key, []))[:max_entities]
                sent_tree.append((token, entities))
                all_tokens.append(token)
                all_entities.append(entities)

                if token in self.special_tags:
                    token_pos_idx = [pos_idx+1]
                    token_abs_idx = [abs_idx+1]
                else:
                    token_pos_idx = [pos_idx+i for i in range(1, len(token)+1)]
                    token_abs_idx = [abs_idx+i for i in range(1, len(token)+1)]
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities:
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent)+1)]
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + i for i in range(1, len(ent)+1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx

            #  combine word group
            for count,sent in enumerate(sent_tree):
                if count == 0:
                    continue
                else:
                    if (all_entities[count] == all_entities[count-1]) and all_entities[count] :
                        all_entities.pop(count)
                        all_tokens[count-1]= all_tokens[count-1]+all_tokens[count]
                        all_tokens.pop(count)
                        sent_tree[count-1][0] = '' +  sent_tree[count-1][0] + ' ' + sent_tree[count][0]
                        sent_tree.pop(count)
                
                '''
            

######################################################################################
            token_group = []
            keys_to_match = []
            in_matching = False
            for token in split_sent:


                if not in_matching:
                    keys_to_match = [x for x in self.lookup_table.keys() if x.startswith(token)]

                if keys_to_match: # start matching
                    token_group.append(token)
                    in_matching = True
                    keys_to_match = [x for x in keys_to_match if token in x]

                    if ([x for x in keys_to_match if x.endswith(token)]) or (not keys_to_match):
                        in_matching = False
                        keys_to_match = []
                        token = ' '.join(token_group)
                        token_group= []

                if in_matching: continue

                if len(keys_to_match) >1:
                    key = keys_to_match[0]
                else: key = token
                entities = list(self.lookup_table.get(key, []))[:max_entities]
                sent_tree.append((token, entities))
                all_tokens.append(token)
                all_entities.append(entities)

                if token in self.special_tags:
                    token_pos_idx = [pos_idx+1]
                    token_abs_idx = [abs_idx+1]
                else:
                    token_pos_idx = [pos_idx+i for i in range(1, len(token.split(' '))+1)]
                    token_abs_idx = [abs_idx+i for i in range(1, len(token.split(' '))+1)]
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities:
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent.split(' '))+1)]
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + i for i in range(1, len(ent.split(' '))+1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx




######################################################################################


            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                if word in self.special_tags:
                    know_sent += [word]
                    seg += [0]
                else:
                    if self.Using_pkuseg:
                        add_word = list(word)
                    else:
                        add_word = [word]
                    know_sent += add_word 
                    seg += [0] * len(' '.join(add_word).split(' '))
                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][1])):
                    add_word = list(sent_tree[i][1][j].split(' '))
                    know_sent += add_word
                    seg += [1] * len(add_word)
                    pos += list(pos_idx_tree[i][1][j])
            
            know_sent = ' '.join(know_sent).split(' ')
            token_num = len(know_sent)

            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1

            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length -  src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]
            
            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)
        
        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch
