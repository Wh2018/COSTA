from base64 import encode
from re import A
from this import d
from turtle import forward
from matplotlib.pyplot import hist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math, copy, time
from torch.autograd import Variable
from dataloader.AVSD_utils_ViT import *
from time import sleep
from transformers import CLIPTextModel, CLIPTokenizer

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#Layer Normalization
class LayerNorm(nn.Module):
    "Construct a layernorm module"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

#Residual connection with layer norm
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size"
    
        return x + self.dropout(sublayer(self.norm(x)))

    def expand_forward(self, x, sublayer):
        out = self.dropout(sublayer(self.norm(x)))
        out = out.mean(1).unsqueeze(1).expand_as(x)
        return x + out 

    def nosum_forward(self, x, sublayer):
        return self.dropout(sublayer(self.norm(x)))


#Gumbel-Softmax sampling
def inverse_gumbel_cdf(y, mu, beta):
    return mu - beta * np.log(-np.log(y))

def gumbel_softmax_sampling(p, mu=0, beta=1, tau=0.1):
    """
    p : (batch_size x 8) tensor. the propability of Q and K
    """
    shape_p = p.shape
    #print('Size-shape_p: ', shape_p)
    y = torch.rand(shape_p) + 1e-25  # ensure all y is positive, batch_size * 8
    #print('Size-y: ', y.size())
    g = (inverse_gumbel_cdf(y, mu, beta)).cuda()
    x = torch.log(p) + g  # samples follow Gumbel distribution.
    # using softmax to generate one_hot vector:
    x = x/tau
    x = F.softmax(x, dim=1)  # now, the x approximates a one_hot vector.

    index = torch.argmax(x, dim=1)
    return index

#Top-k feature selection
def topk_sele(query, key, value, mask=None, dropout=None, top_k=2):

    d_k = query.size(-1)
    query = query.unsqueeze(1)
    value = value.reshape(value.size(0), 8, 4, value.size(2), value.size(3))

    
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    p_attn = F.softmax(scores, dim = -1)

    p_attn = p_attn.squeeze(1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    for i in range(top_k):
        if i == 0:
            index = gumbel_softmax_sampling(p_attn)
            top_seg = torch.cat([value[batch][index[batch].unsqueeze(0)] for batch in range(index.size(0)) ], dim=0).unsqueeze(1)

        else:
            index = gumbel_softmax_sampling(p_attn)

            top_seg_ = torch.cat([value[batch][index[batch].unsqueeze(0)] for batch in range(index.size(0)) ], dim=0).unsqueeze(1)
            top_seg = torch.cat((top_seg, top_seg_), dim=1)
            
    return top_seg

#Coarse-grained Temporal Filtering Step
class Seg_Sele(nn.Module):
    def __init__(self, d_model=512, top_k=2, dropout=0.1):
        super(Seg_Sele, self).__init__()
        self.W_q = nn.Linear(512, d_model)
        self.W_s = nn.Linear(512, d_model)
        self.dropout = nn.Dropout(dropout)
        self.top_k = top_k
        
    def forward(self, query=None, visual=None, query_mask=None):
        
        #frame first token pooling
        frame_fea = visual[:,:,0,:] 
     
        #mean pooling over frame
        seg_fea = torch.cat([torch.mean(frame_fea[:,i*4:i*4+4,:], dim=1, keepdim=True) for i in range(8) ], dim=1)
        
        #question first token pooling
        que_fea = query
        
        Q = self.dropout(self.W_q(que_fea))
        K = self.dropout(self.W_s(seg_fea))

        topk_seg = topk_sele(Q, K ,visual, mask=query_mask, dropout=self.dropout, top_k=self.top_k)
        
        #print('Segment Selection Finished -----------------------')
        return topk_seg, seg_fea

#Top-j region selection
def topj_sele(query, key, value, mask=None, dropout=None, top_j=12):
    d_k = query.size(-1)
    query = query.unsqueeze(1)
    
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    p_attn = F.softmax(scores, dim = -1)

    p_attn = p_attn.squeeze(1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    
    for i in range(top_j):
        if i == 0:
            index = gumbel_softmax_sampling(p_attn)
            top_reg = torch.cat([value[batch][index[batch].unsqueeze(0)] for batch in range(index.size(0)) ], dim=0).unsqueeze(1)
        else:
            index = gumbel_softmax_sampling(p_attn)
            top_reg_ = torch.cat([value[batch][index[batch].unsqueeze(0)] for batch in range(index.size(0)) ], dim=0).unsqueeze(1)
            top_reg = torch.cat((top_reg, top_reg_), dim=1)
            
    return top_reg

#Fine-grained Spatial Filtering Step
class Reg_Sele(nn.Module):
    def __init__(self, d_model=512, top_j=12, dropout=0.1):
        super(Reg_Sele, self).__init__()
        self.W_q = nn.Linear(512, d_model)
        self.W_x = nn.Linear(512, d_model)
        self.dropout = nn.Dropout(dropout)
        self.top_j = top_j
    
    def forward(self, query = None, topk_seg = None, query_mask=None):
        topk_seg = topk_seg.reshape(topk_seg.size(0), topk_seg.size(1) * topk_seg.size(2), topk_seg.size(3), topk_seg.size(4))
        
        #question first token pooling
        que_fea = query
        
        Q = self.dropout(self.W_q(que_fea))
        for frame_idx in range(topk_seg.size(1)):
            frame_fea = topk_seg[:,frame_idx,:,:]
            K = self.dropout(self.W_x(frame_fea))
            
            if frame_idx == 0:
                topj_region = topj_sele(Q, K ,frame_fea, mask=query_mask, dropout=self.dropout, top_j=self.top_j).unsqueeze(1)
            else:
                topj_region_ = topj_sele(Q, K ,frame_fea, mask=query_mask, dropout=self.dropout, top_j=self.top_j).unsqueeze(1)
                topj_region = torch.cat((topj_region, topj_region_), dim=1)
                
        #print('Region Selection Finished -----------------------')
        return topj_region

#Video Positional Encoding
class VideoPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_frames, max_patches, dropout=0.1):
        super(VideoPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Calculate positional encodings for frames
        self.frame_pe = self._get_encoding(max_frames, d_model)
        
        # Calculate positional encodings for patches
        self.patch_pe = self._get_encoding(max_patches, d_model)

    def _get_encoding(self, max_len, d_model):
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        encoding = torch.zeros(1, max_len, d_model)
        encoding[0, :, 0::2] = torch.sin(position * div_term)
        encoding[0, :, 1::2] = torch.cos(position * div_term)
        return encoding

    def forward(self, x):
        batch_size, frames, patches, d_model = x.size()
        
        # Add positional encoding for frames
        x = x + self.frame_pe[:, :frames, :].unsqueeze(2).to(x.device)
        
        # Reshape for adding positional encoding for patches
        x = x.view(batch_size * frames, patches, d_model)
        x = x + self.patch_pe[:, :patches, :].to(x.device)
        
        # Reshape back to the original shape
        x = x.view(batch_size, frames, patches, d_model)
        
        return self.dropout(x)

#Text encoder
class TextEncoder(torch.nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()

        # Load CLIP pre-trained model
        self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")

    def forward(self, inputs):
        # Encode text with CLIP text encoder
        outputs = self.clip_model(**inputs)

        # Extract last hidden state as the text embedding
        text_embedding = outputs.last_hidden_state
        text_pooled = outputs.pooler_output
        return text_embedding, text_pooled
    
# Memory Distillation-inspired Iterative Visual-textural Cross-attention
class ITAM(nn.Module):
    def __init__(self, seg_selec, reg_selec, cross_attn, dis_attn, gate_attn, d_model):
        super().__init__()
        self.seg_selec = seg_selec
        self.reg_selec = reg_selec
        self.cross_attn = cross_attn
        self.dis_attn = dis_attn
        self.gate_attn = gate_attn

        
    def memory_dist(self, visual, X_cro):
        patch_num = visual.size(1) 
        reg_num = visual.size(2)        
        visual = visual.reshape(visual.size(0), patch_num*reg_num, visual.size(3))
     
        input_cat = torch.cat([visual, X_cro], 1)
        input_cat_mask = torch.ones(input_cat.size(0), 1, input_cat.size(1)).cuda()
     
        memory = self.dis_attn(input_cat, input_cat, input_cat, input_cat_mask)
        memory = memory[:, :visual.size(1),:] 
        
        gate = self.gate_attn(input_cat, input_cat, input_cat, input_cat_mask)
        gate  = gate[:, :visual.size(1),:] 
        
        updated_visual = visual * gate + memory * (1 - gate)
        updated_visual = updated_visual.reshape(updated_visual.size(0), patch_num, reg_num, updated_visual.size(2))
   
        return updated_visual
    
    def forward(self, query_emb, query_mask, query_pooled, visual):
        topk_seg, seg_fea = self.seg_selec(query_pooled, visual)
        topj_reg = self.reg_selec(query_pooled, topk_seg)
        
        topj_reg = topj_reg.reshape(topj_reg.size(0), topj_reg.size(1)*topj_reg.size(2), topj_reg.size(3))
        
        #visual-textual cross-modal features
        visual_F = torch.cat((seg_fea, topj_reg),1)

        #cross-attention
        X_cro = self.cross_attn(visual_F, query_emb, query_emb, query_mask)
        
        # memory distillation
        updated_visual = self.memory_dist(visual, X_cro)

        return X_cro, updated_visual
        
#Transformer Encoder
class Encoder(nn.Module):
    def __init__(self, layer ,N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
    
    def forward(self, seq, seq_mask):
        encoded_seq = seq
        for layer in self.layers:
            encoded_seq =  layer(encoded_seq, seq_mask)

        return encoded_seq

#Transformer Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, ff1, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.ff1 = ff1
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size 

    def forward(self, seq, seq_mask):
        #Multi-head Attention->Layer Norm->FFN->layer Norm
        seq = self.sublayer[0](seq, lambda seq: self.self_attn(seq, seq, seq, seq_mask))
        return self.sublayer[1](seq, self.ff1)


#Transformer Decoder
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)


    def forward(self, x, X_ite, X_ite_mask, q_memory, q_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, X_ite, X_ite_mask, q_memory, q_mask, tgt_mask)
        return self.norm(x)

#Transformer Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, size, ans_attn, vis_attn, que_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        
        self.ans_attn = ans_attn
        self.vis_attn = vis_attn
        self.que_attn = que_attn

        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 4)

    def forward(self, x, X_ite, X_ite_mask, q_memory, q_mask, tgt_mask):
        count = 0
        #Answer->History->Captiion->Segment->Region->Question->FF
        
        #Masked answer self-attention
    
        x = self.sublayer[count](x, lambda x: self.ans_attn(x, x, x, tgt_mask))
        count += 1
        
        #visual self-attention
        x = self.sublayer[count](x, lambda x: self.vis_attn(x, X_ite, X_ite, X_ite_mask))
        count += 1
        
        #Query self-attention
        x = self.sublayer[count](x, lambda x: self.que_attn(x, q_memory, q_memory, q_mask))
        count += 1

        #return self.sublayer[count](x, self.feed_forward), out_ae_fts
        return self.sublayer[count](x, self.feed_forward)

#
class EncoderDecoder(nn.Module):
    def __init__(self, visual_ff, vis_pos_emb, ita_mem, text_encoder, decoder, generator):
        super(EncoderDecoder, self).__init__() 
        self.visual_ff = visual_ff
        self.vis_pos_emb = vis_pos_emb
        self.ita_mem = ita_mem
        self.text_encoder = text_encoder
        self.decoder = decoder
        self.generator = generator

    def encode(self, query, visual, query_mask):
        query_emb, query_pooled = self.text_encoder(query)
        visual = self.vis_pos_emb(self.visual_ff(visual)) #batch_size * 32 * 50 *512
        
        updated_visual = visual
        
        # perform iterative cross-attention steps
        X_cro_list = []
        for ita in self.ita_mem:
            X_cro, updated_visual = ita(query_emb, query_mask, query_pooled, updated_visual)
            X_cro_list.append(X_cro)
        
        # cross-modal features mean pooling
        X_ite = torch.mean(torch.stack([x_cro for x_cro in X_cro_list], dim=0), dim=0)
        X_ite_mask = torch.ones(X_ite.size(0), 1, X_ite.size(1)).cuda()
    
        return X_ite, X_ite_mask, query_emb
    

    def decode(self, X_ite, X_ite_mask, q_memory, q_mask, tgt, tgt_mask):
        tgt, _ = self.text_encoder(tgt)
        decoder_output = self.decoder(tgt, X_ite, X_ite_mask, q_memory, q_mask, tgt_mask)

        return decoder_output

    def forward(self, b): 
        #seg_fea, topj_reg, seg_mask, reg_mask = self.encode(b.query_pooled, b.vis_fea)
        X_ite, X_ite_mask, query_emb = self.encode(b.query, b.vis_fea, b.query_mask)
        
        decode_output = self.decode(X_ite, X_ite_mask, query_emb, b.query_mask, b.trg, b.trg_mask)
        
        return decode_output

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
       
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, d_in=-1, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        if d_in < 0: 
            d_in = d_model 
        self.linears = clones(nn.Linear(d_in, d_model), 3)
        self.linears.append(nn.Linear(d_model, d_in))
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1, d_out=-1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        if d_out < 0:
            d_out = d_model 
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class StPositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=50):
        super(StPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
            
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
  
    def forward(self, x, x_st):
        x = x + Variable(self.pe[:, x_st], requires_grad=False)
        x = x.squeeze(0)
        return self.dropout(x)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def make_AVSD_transformer_model(src_vocab, tgt_vocab, batch_size, N=2, d_model=512, d_ff=2048, h=8, dropout=0.1, top_k=2, top_j=12, T=2): 
    c = copy.deepcopy

    visual_ff = nn.Linear(768, d_model)
    vis_pos_em = VideoPositionalEncoding(d_model, 32, 50)
    
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    seg_selec = Seg_Sele(d_model=d_model, top_k=top_k, dropout=dropout)
    reg_selec = Reg_Sele(d_model=d_model, top_j=top_j, dropout=dropout)

    attn = MultiHeadedAttention(h=h, d_model=d_model, dropout=dropout)
    
    #Decoder attention, masked answer->history->caption->segment feature->region feature->query->Liner
    ans_attn = c(attn)
    vis_attn = c(attn)
    que_attn = c(attn)
    cross_attn = c(attn)
    dis_attn = c(attn)
    gate_attn = c(attn)
    
    ita_mem = [ITAM(seg_selec, reg_selec, cross_attn, dis_attn, gate_attn, d_model)]
    for _ in range(T-1):
        ita_mem.append(
            ITAM(seg_selec, reg_selec, cross_attn, dis_attn, gate_attn, d_model)
        )
    ita_mem = nn.ModuleList(ita_mem)
    
    text_encoder = TextEncoder()
    decoder = Decoder(DecoderLayer(d_model, ans_attn, vis_attn, que_attn, c(ff), dropout), N)
    
    generator=Generator(d_model, tgt_vocab)
    
    model = EncoderDecoder(
            visual_ff=visual_ff,
            vis_pos_emb=vis_pos_em,
            ita_mem = ita_mem,
            text_encoder = text_encoder,
            decoder=decoder,
            generator=generator,
          )

    for p in model.parameters():
        if p.dim() > 1:
            #nn.init.xavier_uniform(p)
            nn.init.xavier_uniform_(p)

    return model