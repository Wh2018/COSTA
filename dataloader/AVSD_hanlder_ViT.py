import copy
from json.tool import main
import logging
import sys
import time
import os
import six
import pickle
import json
import numpy as np
import pdb 
import torch 
#from AVSD_utils import * 
from .AVSD_utils_ViT import * 
from time import sleep
from tqdm import tqdm
import re


def end_with_point(str):
    text = re.compile(r".*[0-9a-zA-Z]$")
    if text.match(str):
        return str + ' .'
    else:
        return str


def load_AVSD(dataset_file, include_caption='none', separate_caption=False, max_history_length=-1, merge_source=False, undisclosed_only=False):

    dialog_data = json.load(open(dataset_file, 'r'))
    dialog_list = []
    
    #video id
    vid_set = set()
    qa_id = 0
    for dialog in dialog_data['dialogs']:

        if include_caption == 'caption' or include_caption == 'summary':
            caption = dialog[include_caption]
        elif include_caption == 'caption,summary':
            caption = dialog['caption'] + dialog['summary']
        else:
            caption = []

        caption = end_with_point(caption)
        
        #question and answer->QA pair
        questions = [end_with_point(d['question']) for d in dialog['dialog']]
        answers = [end_with_point(d['answer']) for d in dialog['dialog']]
        qa_pair = [ q + ' ' + a for q,a in zip(questions, answers)]

        vid = dialog['image_id']
        vid_set.add(vid)

        #for test
        if undisclosed_only:
            it = range(len(questions)-1,len(questions))
        else:
            it = range(len(questions))

        #len(it) qa pair
        for n in it:
            if undisclosed_only:
                assert dialog['dialog'][n]['answer'] == '__UNDISCLOSED__'
            if (include_caption == 'caption' or include_caption == 'summary' or include_caption == 'caption,summary') and separate_caption:
                history = ''
            else:
                history = caption
               
            if max_history_length > 0: 
                start_turn_idx = max(0, n - max_history_length)
            else:
                start_turn_idx = 0 

            for m in range(n-1, start_turn_idx-1, -1):
                if history == '':
                    history = qa_pair[m]
                else:
                    history = history + ' ' + qa_pair[m]

            question = questions[n]

            if merge_source:
                question = np.concatenate((caption, history, question))

            answer = answers[n]

            item = [vid, qa_id, history, question, answer]
            if (include_caption == 'caption' or include_caption == 'summary' or include_caption == 'caption,summary') and separate_caption:
                item.append(caption)
            
            dialog_list.append(item)
            qa_id += 1
    data = {'dialogs': dialog_list, 'app_features': [], 'mot_features':[] ,'original': dialog_data}
    
    return data 

def make_batch_indices(data, batchsize=64, max_length=256, separate_caption=0, stage='train'):
    # Setup mini-batches
    idxlist = []

    for n, dialog in enumerate(data['dialogs']):
        vid = dialog[0]  # video ID

        x_len = 0

        qa_id = dialog[1]  # QA-pair id
        h_len = len(dialog[2].split(' '))
        q_len = len(dialog[3].split(' '))
        a_len = len(dialog[4].split(' '))
  
        if separate_caption:
            c_len = len(dialog[5].split(' '))
            idxlist.append((vid, qa_id, x_len, h_len, q_len, a_len, c_len))
        else:
            idxlist.append((vid, qa_id, x_len, h_len, q_len, a_len))
 
    if batchsize > 1:
        if separate_caption:
            idxlist = sorted(idxlist, key=lambda s:(-s[3],-s[6],-s[4],-s[5]))
        else:
            idxlist = sorted(idxlist, key=lambda s:(-s[3],-s[4],-s[5]))

    n_samples = len(idxlist)
    batch_indices = []
    bs = 0
    
    while bs < n_samples:
        #in_len = idxlist[bs][3]
        #bsize = int(batchsize / int(in_len / max_length + 1))
        #be = min(bs + bsize, n_samples) if bsize > 0 else bs + 1
        be = min(bs+batchsize, n_samples)
        if be == n_samples:
            break
        x_len = 0
        h_len = max(idxlist[bs:be], key=lambda s:s[3])[3]
        q_len = max(idxlist[bs:be], key=lambda s:s[4])[4]
        a_len = max(idxlist[bs:be], key=lambda s:s[5])[5]
        
        if separate_caption:#1
            c_len = max(idxlist[bs:be], key=lambda s:s[6])[6]
        vids = [ s[0] for s in idxlist[bs:be] ]
        qa_ids = [ s[1] for s in idxlist[bs:be] ]

        # index[0]: video ids 
        # index[1]: question-answer ids 
        # index[2]: length of video frame sequences for each feature type
        # index[3]: max length of the dialogue history 
        # index[4]: max length of questions
        # index[5]: max length of answers
        # index[-1]: number of dialogues
        if separate_caption:#1
            batch_indices.append((vids, qa_ids, x_len, h_len, q_len, a_len, c_len, be - bs))
        else:
            batch_indices.append((vids, qa_ids, x_len, h_len, q_len, a_len, be - bs))
        bs = be

    return batch_indices, n_samples

def make_batch(data, index, separate_caption=False, skip=[1,1,1], cut_a=False, cut_a_p=0.5, text_tokenizer=None, vis_path=None):
    if separate_caption:
        x_len, h_len, q_len, a_len, c_len, n_seqs = index[2:]
    else:
        x_len, h_len, q_len, a_len, n_seqs = index[2:]

    h_batch = []
    q_batch = []
    a_batch = []
    c_batch = None
    vis_batch = []
    if separate_caption: #1
        c_batch = [] 
    h_st_batch = None
    dialogs = data['dialogs']

    for i in six.moves.range(n_seqs):
        vid = index[0][i]
        
        vis_fea_path = vis_path + vid + '.npy'
        vis_fea = np.load(vis_fea_path)
        vis_batch.append(vis_fea)

        qa_id = index[1][i]
        history, question, answer = dialogs[qa_id][2:5]
    
        if separate_caption:#1
            c_batch.append(dialogs[qa_id][5])
        h_batch.append(history)
    
        q_batch.append(question)
        a_batch.append(answer)

    h_batch = text_tokenizer(h_batch, padding=True, return_tensors="pt", truncation=True).to("cuda")
    h_batch_mask = h_batch['attention_mask'].unsqueeze(1).cuda()

    q_batch = text_tokenizer(q_batch, padding=True, return_tensors="pt", truncation=True).to("cuda")
    q_batch_mask = q_batch['attention_mask'].unsqueeze(1).cuda()

    a_batch = text_tokenizer(a_batch, padding=True, return_tensors="pt", truncation=True).to("cuda")
    trg_batch_for_loss = a_batch['input_ids'].cuda()
    a_batch_mask = a_batch['attention_mask'].unsqueeze(1).cuda()

    
    c_batch = text_tokenizer(c_batch, padding=True, return_tensors="pt", truncation=True).to("cuda")
    c_batch_mask = c_batch['attention_mask'].unsqueeze(1).cuda()
    
    vis_batch = torch.from_numpy(np.asarray(vis_batch)).cuda()
    
    batch = Batch(q_batch, q_batch_mask, h_batch, h_batch_mask, vis_batch, a_batch, a_batch_mask, c_batch, c_batch_mask, trg_batch_for_loss)
    return batch

        