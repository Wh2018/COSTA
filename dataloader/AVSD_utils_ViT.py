import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math, copy, time
import pdb 
from torchtext import data, datasets 
from time import sleep


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, query, query_mask, his, his_mask, vis_fea, trg, trg_mask, cap, cap_mask, trg_batch_for_loss) -> None:
        
        self.query = query
        #self.query_pooled = query_pooled
        self.query_mask = query_mask
        self.his = his
        self.his_mask = his_mask
        self.query_mask = query_mask
        self.cap = cap
        self.cap_mask = cap_mask
        
        self.vis_fea = vis_fea
        if trg is not None:
            self.trg = trg
            self.trg_mask = trg_mask
            self.trg_batch_for_loss = trg_batch_for_loss
            #self.ntokens = 1##
            self.ntokens = (self.trg_mask != 0).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) * \
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None, l=1.0):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt 
        self.l = l
    
    def __call__(self, x, y, norm, ae_x=None, ae_y=None, ae_norm=None):
        out = self.generator(x)
        loss = self.criterion(out.contiguous().view(-1, out.size(-1)), 
                              y.contiguous().view(-1)) / norm.float()
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm.float()

def beam_search_decode(model, batch, max_len, start_symbol, unk_symbol, end_symbol, pad_symbol, beam=5, penalty=1.0, nbest=5, min_len=1):
    
    X_ite, X_ite_mask, query_emb = model.encode(batch.query, batch.vis_fea, batch.query_mask)

    ds = torch.ones(1, 1).fill_(start_symbol).type_as(query_emb.data)
    hyplist=[([], 0., ds)]
    best_state=None
    comp_hyplist=[]
    for l in range(max_len): 
        new_hyplist = []
        argmin = 0
        for out, lp, st in hyplist:
            
            cap2res_mask = None
            output = model.decode(X_ite, X_ite_mask, query_emb, batch.query_mask,
                                     Variable(st), Variable(subsequent_mask(st.size(1)).type_as(query_emb.data)))

            if type(output) == tuple or type(output) == list:
                logp = model.generator(output[0][:, -1])
            else:
                logp = model.generator(output[:, -1])
            lp_vec = logp.cpu().data.numpy() + lp 
            lp_vec = np.squeeze(lp_vec)
            if l >= min_len:
                new_lp = lp_vec[end_symbol] + penalty * (len(out) + 1)
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state < new_lp: 
                    best_state = new_lp
            count = 1 
            for o in np.argsort(lp_vec)[::-1]:
                if o == unk_symbol or o == end_symbol:
                    continue 
                new_lp = lp_vec[o]
                if len(new_hyplist) == beam:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = torch.cat([st, torch.ones(1,1).type_as(batch.query.data).fill_(int(o))], dim=1)
                        new_hyplist[argmin] = (out + [o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
                    else:
                        break
                else: 
                    new_st = torch.cat([st, torch.ones(1,1).type_as(query_emb.data).fill_(int(o))], dim=1)
                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == beam:
                        argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
                count += 1
        hyplist = new_hyplist 
            
    if len(comp_hyplist) > 0: 
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]
        return maxhyps, best_state
    else:
        return [([], 0)], None

def greedy_decode(model, batch, max_len, start_symbol, pad_symbol):
    X_ite, X_ite_mask, query_emb = model.encode(batch.query, batch.vis_fea, batch.query_mask)
    
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(query_emb.data)

    for i in range(max_len-1):
        out = model.decode(X_ite, X_ite_mask, query_emb, batch.query_mask,
                                     Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(query_emb.data)))
        if type(out) == list:
            prob = 0
            for idx, o in enumerate(out):
                prob += model.generator[idx](o[:,-1])
        else:
            prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(query_emb.data).fill_(next_word)], dim=1)
    return ys