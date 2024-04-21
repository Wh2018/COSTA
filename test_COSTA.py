#

import argparse
import logging
import math
import sys
import time
import os
import copy
import pickle
import json

import numpy as np
import six

import torch
import torch.nn as nn
import dataloader.AVSD_hanlder_ViT as AVSD
from dataloader.AVSD_utils_ViT import * 
from transformers import AutoTokenizer, CLIPTextModel

# Evaluation routine
def generate_response(model, data, batch_indices, vocab, maxlen=20, beam=5, penalty=2.0, nbest=1, ref_data=None, text_tokenizer=None, v_fea_dir=None):
    vocablist = sorted(vocab.keys(), key=lambda s:vocab[s])
    result_dialogs = []
    model.eval()
    with torch.no_grad():
        qa_id = 0
        for idx, dialog in enumerate(data['original']['dialogs']):
            vid = dialog['image_id']
            if args.undisclosed_only:
                out_dialog = dialog['dialog'][-1:]
                if ref_data is not None:
                    ref_dialog = ref_data['dialogs'][idx]
                    assert ref_dialog['image_id'] == vid 
                    ref_dialog = ref_dialog['dialog'][-1:]
            else:
                out_dialog = dialog['dialog']
            pred_dialog = {'image_id': vid,
                           'dialog': copy.deepcopy(out_dialog)}
            result_dialogs.append(pred_dialog)
            for t, qa in enumerate(out_dialog):
                if args.undisclosed_only:
                    assert qa['answer'] == '__UNDISCLOSED__'
                logging.info('%d %s_%d' % (qa_id, vid, t))
                logging.info('QS: ' + qa['question'])
                if args.undisclosed_only and ref_data is not None:
                    logging.info('REF: ' + ref_dialog[t]['answer'])
                else:
                    logging.info('REF: ' + qa['answer'])
                # prepare input data
                start_time = time.time()
                batch = AVSD.make_batch(data, batch_indices[qa_id], separate_caption=True, text_tokenizer=text_tokenizer, vis_path=v_fea_dir)
                qa_id += 1
                if args.decode_style == 'beam_search': 
                  pred_out, _ = beam_search_decode(model, batch, maxlen, start_symbol=vocab['<|startoftext|>'], unk_symbol=vocab['<|endoftext|>'], end_symbol=vocab['<|endoftext|>'], pad_symbol=vocab['<|endoftext|>'], beam=beam, penalty=penalty, nbest=nbest)
                  for n in range(min(nbest, len(pred_out))):
                    pred = pred_out[n]
                    hypstr = []
                    for w in pred[0]:
                        if w == vocab['<|endoftext|>']:
                            break
                        hypstr.append(vocablist[w])
                    hypstr = " ".join(hypstr)
                    #hypstr = " ".join([vocablist[w] for w in pred[0]])
                    logging.info('HYP[%d]: %s  ( %f )' % (n + 1, hypstr, pred[1]))
                    if n == 0: 
                        pred_dialog['dialog'][t]['answer'] = hypstr
                elif args.decode_style == 'greedy': 
                  output = greedy_decode(model, batch, maxlen, start_symbol=vocab['<|startoftext|>'], pad_symbol=vocab['<|endoftext|>'])
                  output = [i for i in output[0].cpu().numpy()]
                  hypstr = []
                  for i in output[1:]:
                    if i == vocab['<|endoftext|>']:
                        break
                    hypstr.append(vocablist[i])
                  hypstr = ' '.join(hypstr)
                  logging.info('HYP: {}'.format(hypstr))
                  pred_dialog['dialog'][t]['answer'] = hypstr
                logging.info('ElapsedTime: %f' % (time.time() - start_time))
                logging.info('-----------------------')

    return {'dialogs': result_dialogs}


##################################
# main
if __name__ =="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--test_path', default='', type=str,
                        help='Path to test feature files')
    parser.add_argument('--test_set', default='data/AVSD/test_set4DSTC7-AVSD.json', type=str,
                        help='Filename of test data')
    parser.add_argument('--model_conf', default='out_models/COSTA/.conf', type=str,
                        help='Attention model to be output')
    parser.add_argument('--include_caption', default='caption', type=str, help='Include caption in the history')
    parser.add_argument('--separate_caption', default=1, type=int, help='Separate caption from dialogue history')
    parser.add_argument('--model', '-m', default='out_models/COSTA/_best', type=str,
                        help='Attention model to be output')
    parser.add_argument('--maxlen', default=20, type=int,
                        help='Max-length of output sequence')
    parser.add_argument('--beam', default=5, type=int,
                        help='Beam width')
    parser.add_argument('--penalty', default=1.0, type=float,
                        help='Insertion penalty')
    parser.add_argument('--nbest', default=5, type=int,
                        help='Number of n-best hypotheses')
    parser.add_argument('--output', '-o', default='out_models/COSTA/generated_results.json', type=str,
                        help='Output generated responses in a json file')
    parser.add_argument('--verbose', '-v', default=0, type=int,
                        help='verbose level')
    parser.add_argument('--decode_style', default='beam_search', type=str, help='greedy or beam_search')
    parser.add_argument('--undisclosed_only', default=1, type=int, help='')
    parser.add_argument('--labeled_test', default='data/AVSD/lbl_test_set4DSTC7-AVSD.json', type=str, help='directory to labelled data')
    parser.add_argument('--v_fea_dir', default='data/Charades_ViT-B16-test/', type=str,help='The folder where video features are stored ')

    args = parser.parse_args()
    args.undisclosed_only = bool(args.undisclosed_only)
    for arg in vars(args):
        print("{}={}".format(arg, getattr(args, arg)))

    if args.verbose >= 1:
        logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s')
 
    logging.info('Loading model params from ' + args.model)
    path = args.model_conf
    with open(path, 'rb') as f:
        vocab, train_args = pickle.load(f)
    model = torch.load(args.model+'.pth.tar')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    text_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")

    
    # report data summary
    logging.info('#vocab = %d' % len(vocab))
    # prepare test data
    logging.info('Loading test data from ' + args.test_set)
    test_data = AVSD.load_AVSD(dataset_file=args.test_set, 
                        include_caption=args.include_caption,
                        separate_caption=args.separate_caption,
                        max_history_length=train_args.max_history_length,
                        merge_source=train_args.merge_source,
                        undisclosed_only=args.undisclosed_only)
    test_indices, test_samples = AVSD.make_batch_indices(test_data, 1, separate_caption=args.separate_caption,stage='test')
    logging.info('#test sample = %d' % test_samples)
    logging.info('#test batch = %d' % len(test_indices))
    # generate sentences
    logging.info('-----------------------generate--------------------------')
    start_time = time.time()
    labeled_test = None 
    if args.undisclosed_only and args.labeled_test is not None:
        labeled_test = json.load(open(args.labeled_test, 'r'))
    result = generate_response(model, test_data, test_indices, vocab, 
                               maxlen=args.maxlen, beam=args.beam, 
                               penalty=args.penalty, nbest=args.nbest, ref_data=labeled_test,
                                text_tokenizer=text_tokenizer, v_fea_dir=args.v_fea_dir)
    logging.info('----------------')
    logging.info('wall time = %f' % (time.time() - start_time))
    if args.output:
        logging.info('writing results to ' + args.output)
        json.dump(result, open(args.output, 'w'), indent=4)
    logging.info('done')
