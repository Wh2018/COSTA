import argparse
import logging
import math

import sys
import time
import random
import os
import json
import numpy as np
import pickle as pkl
import threading
from tqdm import tqdm 
from time import sleep
import torch
import torch.nn as nn
import dataloader.AVSD_hanlder_ViT as AVSD
from lib.label_smoothing import *
from lib.ViT_transformer import *
#from dataloader.AVSD_utils import * 
from transformers import AutoTokenizer, CLIPTextModel

def run_epoch(data, indices, vocab, epoch, model, loss_compute, eval=False, text_tokenizer=None,v_fea_dir=None):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0  
    total_loss = 0 
    tokens = 0
    it = tqdm(range(len(indices)), desc="epoch {}/{}".format(epoch+1, args.num_epochs), ncols=0)
    for j in it:

        batch = AVSD.make_batch(data, indices[j], separate_caption=args.separate_caption, cut_a=args.cut_a, text_tokenizer=text_tokenizer, vis_path=v_fea_dir)
        b = batch
        out = model.forward(b)

        loss = loss_compute(out, b.trg_batch_for_loss, b.ntokens)

        total_loss += loss
        total_tokens += b.ntokens
        tokens += b.ntokens
        if (j+1) % args.report_interval == 0 and not eval:
            elapsed = time.time() - start
            print("Epoch: %d Step: %d Loss: %f Tokens per Sec: %f" %
                    (epoch+1,j+1, loss / b.ntokens.float(), float(tokens) / elapsed))
            with open(train_log_path, "a") as f:
                f.write("{},{},{:e},{}\n".format(epoch+1,j+1,loss/b.ntokens.float(),float(tokens)/elapsed))
            start = time.time()
            tokens = 0
        #prefetch.join()
    return total_loss / total_tokens.float()


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    # train, dev and test data
    parser.add_argument('--train_set', default='data/AVSD/train_set4DSTC7-AVSD.json', type=str,help='Filename of train data')
    parser.add_argument('--valid_set', default='data/AVSD/valid_set4DSTC7-AVSD.json', type=str,help='Filename of validation data')
    parser.add_argument('--include_caption', default='caption', type=str, help='Include caption in the history')
    parser.add_argument('--separate_caption', default=1, type=int, help='Separate caption from dialogue history')
    parser.add_argument('--cut_a', default=1, type=int, help='randomly cut responses to simulate bs') 
    parser.add_argument('--merge_source', default=0, type=int, help='merge all source sequences into one') 
    parser.add_argument('--exclude_video', action='store_true',help='')
    parser.add_argument('--model', default='out_models/COSTA/', type=str,help='output path of model and params')
    parser.add_argument('--v_fea_dir', default='data/Charades_ViT-B16/', type=str,help='The folder where video features are stored ')
    # Model 
    parser.add_argument('--nb_blocks', default=2, type=int,help='number of transformer blocks')
    parser.add_argument('--d_model', default=512, type=int, help='dimension of model tensors') 
    parser.add_argument('--d_ff', default=2048, type=int, help='dimension of feed forward') 
    parser.add_argument('--att_h', default=8, type=int, help='number of attention heads') 
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--top_k', default=2, type=str, help='top_k for video clip extraction')
    parser.add_argument('--top_j', default=12, type=str, help='top_p for image region extraction')
    parser.add_argument('--T', default=2, type=str, help='iterative times')
    # Training 
    parser.add_argument('--num_epochs', '-e', default=10, type=int,help='Number of epochs')
    parser.add_argument('--rand_seed', '-s', default=1, type=int, help="seed f or generating random numbers")
    parser.add_argument('--batch_size', '-b', default=32, type=int,help='Batch size in training')
    parser.add_argument('--max_length', default=64, type=int,help='Maximum length for controling batch size')
    parser.add_argument('--max_history_length', default=5, type=int, help='Maximum past history length to consider')
    parser.add_argument('--report_interval', default=10, type=int,help='report interval to log training results')
    parser.add_argument('--warmup_steps', default=9660, type=int,help='warm up steps for optimizer') 
    parser.add_argument('--loss_l', default=1.0, type=float, help="")
    # others
    parser.add_argument('--verbose', '-v', default=0, type=int,help='verbose level')

    args = parser.parse_args()
    
    # Presetting
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    if args.verbose >= 1:
        logging.basicConfig(level=logging.DEBUG, 
            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s %(levelname)s: %(message)s')
    for arg in vars(args):
        print("{}={}".format(arg, getattr(args, arg)))
    

    # load train and val data
    logging.info('Loading training data from ' + args.train_set)
    train_data = AVSD.load_AVSD(args.train_set,
                         include_caption=args.include_caption, separate_caption=args.separate_caption,
                         max_history_length=args.max_history_length, 
                         merge_source=args.merge_source)

    logging.info('Loading validation data from ' + args.valid_set)
    valid_data = AVSD.load_AVSD(args.valid_set, 
                         include_caption=args.include_caption, separate_caption=args.separate_caption, 
                         max_history_length=args.max_history_length, 
                         merge_source=args.merge_source)
    
    # report data summary
    logging.info('#train_size = %d,val_size = %d'% (len(train_data['dialogs']),len(valid_data['dialogs'])))
    # make batchset for training
    train_indices, train_samples = AVSD.make_batch_indices(train_data, args.batch_size,
                                                         max_length=args.max_length, separate_caption=args.separate_caption, stage='train')
    logging.info('#train sample = %d' % train_samples)
    logging.info('#train batch = %d' % len(train_indices))
    # make batchset for validation
    valid_indices, valid_samples = AVSD.make_batch_indices(valid_data, args.batch_size,
                                                     max_length=args.max_length, separate_caption=args.separate_caption, stage='train')
    logging.info('#validation sample = %d' % valid_samples)
    logging.info('#validation batch = %d' % len(valid_indices))
    
    text_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    vocab = text_tokenizer.get_vocab()
    

    #create model
    model = make_AVSD_transformer_model(len(vocab), len(vocab), batch_size=args.batch_size, N=args.nb_blocks, d_model=args.d_model, d_ff=args.d_ff, h=args.att_h, dropout=args.dropout, top_k=args.top_k, top_j=args.top_j, T=args.T) 
    model.cuda()

    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    for param in model.parameters():
        mulValue = np.prod(param.size())  
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue 
        else:
            NonTrainable_params += mulValue 
    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')
    

    criterion = LabelSmoothing(size=len(vocab), padding_idx=vocab['<|endoftext|>'], smoothing=0.1)
    criterion.cuda()

    # save meta parameters
    if os.path.exists(args.model) == False:
        os.mkdir(args.model)
    path = args.model + '.conf'
    with open(path, 'wb') as f:
        pkl.dump((vocab, args), f, -1)
    path2 = args.model + '_params.txt'
    with open(path2, "w") as f: 
        for arg in vars(args):
            f.write("{}={}\n".format(arg, getattr(args, arg)))

    
    logging.info('-------------------------')
    logging.info('Start training')
    logging.info('-------------------------')

    # initialize status parameters
    modelext = '.pth.tar'
    min_valid_loss = 1.0e+10
    bestmodel_num = 0

    # save results 
    trace_log_path = args.model+'_trace.csv'
    with open(trace_log_path, "w") as f:
        f.write('epoch,split,avg_loss\n') 
    train_log_path = args.model+'_train.csv'
    with open(train_log_path, "w") as f:  
        f.write('epoch,step,loss,tokens_per_sec\n') 
    print("Saving training results to {}".format(train_log_path))
    print("Saving val results to {}".format(trace_log_path))   


    model_optimizer = NoamOpt(args.d_model, 1, args.warmup_steps,
            torch.optim.AdamW(model.parameters(), lr=0.00005, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(args.num_epochs):
        # start training 
        logging.info('------------training-------------')
        random.shuffle(train_indices)
        model.train()
    
        train_loss = run_epoch(train_data, train_indices, vocab, epoch,
                model,
                SimpleLossCompute(model.generator,
                criterion,opt=model_optimizer, l=args.loss_l), text_tokenizer=text_tokenizer, v_fea_dir=args.v_fea_dir)
        logging.info("epoch: %d  train loss: %f" % (epoch+1, train_loss))

        # test on validation data 
        logging.info('------------validation-------------')
        model.eval()
        #object_detector.is_train= False
        with torch.no_grad():
            valid_loss = run_epoch(valid_data, valid_indices, vocab, epoch,
                model,
                SimpleLossCompute(model.generator,
                criterion,opt=None, l=args.loss_l),
                eval=True, text_tokenizer=text_tokenizer, v_fea_dir=args.v_fea_dir)
        logging.info('epoch: %d validation loss: %f' % (epoch+1, valid_loss))

        with open(trace_log_path,"a") as f:
            f.write("{},train,{:e}\n".format(epoch+1,train_loss))
            f.write("{},val,{:e}\n".format(epoch+1,valid_loss))       

        # update the model and save checkpoints
        modelfile = args.model + '_' + str(epoch + 1) + modelext
        logging.info('writing model params to ' + modelfile)
        torch.save(model, modelfile)
        if min_valid_loss > valid_loss:
            bestmodel_num = epoch+1
            logging.info('validation loss reduced %.4f -> %.4f' % (min_valid_loss, valid_loss))
            min_valid_loss = valid_loss
            logging.info('a symbolic link is made as ' + args.model + '_best' + modelext)
            if os.path.exists(args.model + '_best' + modelext):
                os.remove(args.model + '_best' + modelext)
            os.symlink(os.path.basename(args.model + '_' + str(bestmodel_num) + modelext), args.model + '_best' + modelext)
        logging.info('----------------')
    logging.info('the best model is epoch %d.' % bestmodel_num)