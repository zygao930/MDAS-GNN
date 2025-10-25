import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from lib.utils import norm_Adj

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0   # 1 means reachable; 0 means unreachable

class spatialGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
        super(spatialGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))

class GCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
        super(GCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)))  # (N,N)(b,N,in)->(b,N,in)->(b,N,out)

class FeatureSpecificSpatialAttention(nn.Module):
    def __init__(self, d_model, num_features=3, dropout=0.0):
        super(FeatureSpecificSpatialAttention, self).__init__()
        self.d_model = d_model
        self.num_features = num_features
        self.dropout = nn.Dropout(p=dropout)
        
        # Feature-specific transformation matrices
        self.feature_transforms = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(num_features)
        ])

    def forward(self, x):
        # x: (batch, N, T, d_model)
        batch_size, num_vertices, num_timesteps, d_model = x.shape
        
        attended_features = []
        
        for f in range(self.num_features):
            # Reshape for attention computation: (batch*T, N, d_model)
            x_reshaped = x.permute(0, 2, 1, 3).reshape(-1, num_vertices, d_model)
            
            # Compute feature-specific attention
            scores = torch.matmul(x_reshaped, x_reshaped.transpose(-2, -1)) / math.sqrt(d_model)
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention and feature-specific transformation
            attended = torch.matmul(attention_weights, x_reshaped)
            attended = self.feature_transforms[f](attended)
            
            # Reshape back: (batch, N, T, d_model)
            attended = attended.reshape(batch_size, num_timesteps, num_vertices, d_model).permute(0, 2, 1, 3)
            attended_features.append(attended)
        
        # Average the different feature-specific attention outputs
        return torch.stack(attended_features, dim=0).mean(dim=0)

class Spatial_Attention_layer(nn.Module):
    def __init__(self, dropout=.0):
        super(Spatial_Attention_layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)
        score = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(in_channels)  # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)
        score = self.dropout(F.softmax(score, dim=-1))  # the sum of each row is 1; (b*t, N, N)
        return score.reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices))

class spatialAttentionGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, dropout=.0):
        super(spatialAttentionGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.SAt = Spatial_Attention_layer(dropout=dropout)

    def forward(self, x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        spatial_attention = self.SAt(x)  # (batch, T, N, N)
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)
        spatial_attention = spatial_attention.reshape((-1, num_of_vertices, num_of_vertices))  # (b*T, n, n)
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix.mul(spatial_attention), x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))
 
class spatialAttentionScaledGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, dropout=.0):
        super(spatialAttentionScaledGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.SAt = Spatial_Attention_layer(dropout=dropout)

    def forward(self, x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        spatial_attention = self.SAt(x) / math.sqrt(in_channels)  # scaled self attention: (batch, T, N, N)
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))
        spatial_attention = spatial_attention.reshape((-1, num_of_vertices, num_of_vertices))  # (b*T, n, n)
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix.mul(spatial_attention), x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))

class FeatureSpecificSpatialAttentionGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, dropout=.0):
        super(FeatureSpecificSpatialAttentionGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.feature_spatial_attn = FeatureSpecificSpatialAttention(in_channels, dropout=dropout)

    def forward(self, x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        
        # Apply feature-specific spatial attention
        x_attended = self.feature_spatial_attn(x)
        
        # Standard GCN processing
        x_attended = x_attended.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)
        output = F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x_attended)))
        
        return output.reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2)

class FeatureSpecificSpatialAttentionScaledGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, dropout=.0):
        super(FeatureSpecificSpatialAttentionScaledGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.feature_spatial_attn = FeatureSpecificSpatialAttention(in_channels, dropout=dropout)

    def forward(self, x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        
        # Apply feature-specific spatial attention with scaling
        x_attended = self.feature_spatial_attn(x) / math.sqrt(in_channels)
        
        # Standard GCN processing
        x_attended = x_attended.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)
        output = F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x_attended)))
        
        return output.reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2)

class PositionWiseGCNFeedForward(nn.Module):
    def __init__(self, gcn, dropout=.0):
        super(PositionWiseGCNFeedForward, self).__init__()
        self.gcn = gcn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.gcn(x))

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len, lookup_index=None):
        super(TemporalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.lookup_index = lookup_index

    def forward(self, x):
        if self.lookup_index is not None:
            x = x + self.pe[self.lookup_index, :]
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SpatialPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_of_vertices, dropout, gcn, smooth_layer_num=0):
        super(SpatialPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.gcn_smooth_layers = None
        if smooth_layer_num > 0:
            self.gcn_smooth_layers = nn.ModuleList([gcn for _ in range(smooth_layer_num)])
        pe = torch.randn(num_of_vertices, d_model)
        self.pe = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        if self.gcn_smooth_layers is not None:
            pe = self.pe
            for layer in self.gcn_smooth_layers:
                pe = layer(pe)
            x = x + pe
        else:
            x = x + self.pe
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, nb_head, d_model, dropout=.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, query_multi_segment=True, key_multi_segment=True):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        if query_multi_segment and key_multi_segment:
            query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                               for l, x in zip(self.linears, (query, key, value))]
        elif (not query_multi_segment) and (not key_multi_segment):
            query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                               for l, x in zip(self.linears, (query, key, value))]
        elif (not query_multi_segment) and key_multi_segment:
            query = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            key = self.linears[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            value = self.linears[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttentionAwareTemporalContex_q1d_k1d(nn.Module):
    def __init__(self, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, num_for_predict, kernel_size=3, dropout=.0):
        super(MultiHeadAttentionAwareTemporalContex_q1d_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.linears_q = clones(nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, 1)), 1)
        self.linears_k = clones(nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, 1)), 1)
        self.linears_v = clones(nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, 1)), 1)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.num_of_weeks = num_of_weeks
        self.num_of_days = num_of_days
        self.num_of_hours = num_of_hours
        self.num_for_predict = num_for_predict

    def forward(self, query, key, value, mask=None, query_multi_segment=True, key_multi_segment=True):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        N = query.size(1)
        
        # 1d conv on query, key, value
        query = query.view(nbatches*N, self.num_of_weeks*self.num_for_predict, -1).transpose(1, 2).unsqueeze(-2)
        key = key.view(nbatches*N, (self.num_of_weeks + self.num_of_days + self.num_of_hours)*self.num_for_predict, -1).transpose(1, 2).unsqueeze(-2)
        value = value.view(nbatches*N, (self.num_of_weeks + self.num_of_days + self.num_of_hours)*self.num_for_predict, -1).transpose(1, 2).unsqueeze(-2)

        query = self.linears_q[0](query).squeeze(-2).transpose(1, 2).contiguous().view(nbatches, N, self.num_of_weeks*self.num_for_predict, -1)
        key = self.linears_k[0](key).squeeze(-2).transpose(1, 2).contiguous().view(nbatches, N, (self.num_of_weeks + self.num_of_days + self.num_of_hours)*self.num_for_predict, -1)
        value = self.linears_v[0](value).squeeze(-2).transpose(1, 2).contiguous().view(nbatches, N, (self.num_of_weeks + self.num_of_days + self.num_of_hours)*self.num_for_predict, -1)

        if query_multi_segment and key_multi_segment:
            query, key, value = [l(x).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)
                               for l, x in zip(self.linears[:-1], (query, key, value))]
        elif (not query_multi_segment) and (not key_multi_segment):
            query, key, value = [l(x).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)
                               for l, x in zip(self.linears[:-1], (query, key, value))]
        elif (not query_multi_segment) and key_multi_segment:
            query = self.linears[0](query).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)
            key = self.linears[1](key).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)
            value = self.linears[2](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(2, 3).contiguous().view(nbatches, N, -1, self.h * self.d_k)
        return self.linears[-1](x)

class MultiHeadAttentionAwareTemporalContex_qc_k1d(nn.Module):
    def __init__(self, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, num_for_predict, kernel_size=3, dropout=.0):
        super(MultiHeadAttentionAwareTemporalContex_qc_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.linears_k = clones(nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, 1)), 1)
        self.linears_v = clones(nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, 1)), 1)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.num_of_weeks = num_of_weeks
        self.num_of_days = num_of_days
        self.num_of_hours = num_of_hours
        self.num_for_predict = num_for_predict

    def forward(self, query, key, value, mask=None, query_multi_segment=True, key_multi_segment=True):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        N = query.size(1)

        key = key.view(nbatches*N, (self.num_of_weeks + self.num_of_days + self.num_of_hours)*self.num_for_predict, -1).transpose(1, 2).unsqueeze(-2)
        value = value.view(nbatches*N, (self.num_of_weeks + self.num_of_days + self.num_of_hours)*self.num_for_predict, -1).transpose(1, 2).unsqueeze(-2)

        key = self.linears_k[0](key).squeeze(-2).transpose(1, 2).contiguous().view(nbatches, N, (self.num_of_weeks + self.num_of_days + self.num_of_hours)*self.num_for_predict, -1)
        value = self.linears_v[0](value).squeeze(-2).transpose(1, 2).contiguous().view(nbatches, N, (self.num_of_weeks + self.num_of_days + self.num_of_hours)*self.num_for_predict, -1)

        if query_multi_segment and key_multi_segment:
            query, key, value = [l(x).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)
                               for l, x in zip(self.linears[:-1], (query, key, value))]
        elif (not query_multi_segment) and (not key_multi_segment):
            query, key, value = [l(x).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)
                               for l, x in zip(self.linears[:-1], (query, key, value))]
        elif (not query_multi_segment) and key_multi_segment:
            query = self.linears[0](query).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)
            key = self.linears[1](key).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)
            value = self.linears[2](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(2, 3).contiguous().view(nbatches, N, -1, self.h * self.d_k)
        return self.linears[-1](x)

class MultiHeadAttentionAwareTemporalContex_qc_kc(nn.Module):
    def __init__(self, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, num_for_predict, kernel_size=3, dropout=.0):
        super(MultiHeadAttentionAwareTemporalContex_qc_kc, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.num_of_weeks = num_of_weeks
        self.num_of_days = num_of_days
        self.num_of_hours = num_of_hours
        self.num_for_predict = num_for_predict

    def forward(self, query, key, value, mask=None, query_multi_segment=True, key_multi_segment=True):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        N = query.size(1)

        if query_multi_segment and key_multi_segment:
            query, key, value = [l(x).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)
                               for l, x in zip(self.linears[:-1], (query, key, value))]
        elif (not query_multi_segment) and (not key_multi_segment):
            query, key, value = [l(x).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)
                               for l, x in zip(self.linears[:-1], (query, key, value))]
        elif (not query_multi_segment) and key_multi_segment:
            query = self.linears[0](query).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)
            key = self.linears[1](key).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)
            value = self.linears[2](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(2, 3).contiguous().view(nbatches, N, -1, self.h * self.d_k)
        return self.linears[-1](x)

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout, residual_connection=True, use_LayerNorm=True):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size) if use_LayerNorm else None
        self.dropout = nn.Dropout(dropout)
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm

    def forward(self, x, sublayer):
        if self.residual_connection and self.use_LayerNorm:
            return x + self.dropout(sublayer(self.norm(x)))
        elif self.residual_connection and (not self.use_LayerNorm):
            return x + self.dropout(sublayer(x))
        elif (not self.residual_connection) and self.use_LayerNorm:
            return self.dropout(sublayer(self.norm(x)))
        else:
            return self.dropout(sublayer(x))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout, residual_connection=True, use_LayerNorm=True):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 2)
        self.size = size
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm

    def forward(self, x):
        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, query_multi_segment=True, key_multi_segment=True))
            return self.sublayer[1](x, self.feed_forward)
        else:
            x = self.self_attn(x, x, x, query_multi_segment=True, key_multi_segment=True)
            return self.feed_forward(x)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, DEVICE):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.DEVICE = DEVICE

    def forward(self, src, tgt):
        return self.decode(tgt, self.encode(src))

    def encode(self, src):
        return self.encoder(self.src_embed(src))

    def decode(self, tgt, memory):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory))

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, residual_connection=True, use_LayerNorm=True):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward_gcn = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 3)
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm

    def forward(self, x, m):
        tgt_mask = subsequent_mask(x.size(-2)).type_as(x.data)
        
        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, query_multi_segment=False, key_multi_segment=False))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, query_multi_segment=False, key_multi_segment=True))
            return self.sublayer[2](x, self.feed_forward_gcn)
        else:
            x = self.self_attn(x, x, x, tgt_mask, query_multi_segment=False, key_multi_segment=False)
            x = self.src_attn(x, m, m, query_multi_segment=False, key_multi_segment=True)
            return self.feed_forward_gcn(x)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory):
        for layer in self.layers:
            x = layer(x, memory)
        return self.norm(x)

def search_index(max_len, num_of_depend, num_for_predict, points_per_hour, units):
    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = max_len - points_per_hour * units * i
        for j in range(num_for_predict):
            end_idx = start_idx + j
            x_idx.append(end_idx)
    return x_idx

class SpatialTemporalEmbedding(nn.Module):
    def __init__(self, dense, temporal_pos, spatial_pos):
        super(SpatialTemporalEmbedding, self).__init__()
        self.dense = dense
        self.temporal_pos = temporal_pos
        self.spatial_pos = spatial_pos
        
    def forward(self, x):
        x = self.dense(x)
        x = self.temporal_pos(x)
        x = self.spatial_pos(x)
        return x

class SpatialEmbedding(nn.Module):
    def __init__(self, dense, spatial_pos):
        super(SpatialEmbedding, self).__init__()
        self.dense = dense
        self.spatial_pos = spatial_pos
        
    def forward(self, x):
        x = self.dense(x)
        x = self.spatial_pos(x)
        return x

def make_model(DEVICE, num_layers, encoder_input_size, decoder_output_size, d_model, adj_mx, nb_head, num_of_weeks, num_of_days, 
              num_of_hours, points_per_hour, num_for_predict, dropout=.0, aware_temporal_context=True, ScaledSAt=True, SE=True, 
              TE=True, kernel_size=3, smooth_layer_num=0, residual_connection=True, use_LayerNorm=True):

    c = copy.deepcopy
    norm_Adj_matrix = torch.from_numpy(norm_Adj(adj_mx)).type(torch.FloatTensor).to(DEVICE)
    num_of_vertices = norm_Adj_matrix.shape[0]
    src_dense = nn.Linear(encoder_input_size, d_model)

    # Use feature-specific GCN
    if ScaledSAt:
        position_wise_gcn = PositionWiseGCNFeedForward(FeatureSpecificSpatialAttentionScaledGCN(norm_Adj_matrix, d_model, d_model), dropout=dropout)
    else:
        position_wise_gcn = PositionWiseGCNFeedForward(FeatureSpecificSpatialAttentionGCN(norm_Adj_matrix, d_model, d_model), dropout=dropout)

    trg_dense = nn.Linear(decoder_output_size, d_model)

    # Encoder temporal position embedding
    max_len = max(num_of_weeks * 7 * 24 * num_for_predict, num_of_days * 24 * num_for_predict, num_of_hours * num_for_predict)

    w_index = search_index(max_len, num_of_weeks, num_for_predict, points_per_hour, 7*24)
    d_index = search_index(max_len, num_of_days, num_for_predict, points_per_hour, 24)
    h_index = search_index(max_len, num_of_hours, num_for_predict, points_per_hour, 1)
    en_lookup_index = w_index + d_index + h_index

    if aware_temporal_context:
        attn_ss = MultiHeadAttentionAwareTemporalContex_q1d_k1d(nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, num_for_predict, kernel_size, dropout=dropout)
        attn_st = MultiHeadAttentionAwareTemporalContex_qc_k1d(nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, num_for_predict, kernel_size, dropout=dropout)
        att_tt = MultiHeadAttentionAwareTemporalContex_qc_kc(nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, num_for_predict, kernel_size, dropout=dropout)
    else:
        attn_ss = MultiHeadAttention(nb_head, d_model, dropout=dropout)
        attn_st = MultiHeadAttention(nb_head, d_model, dropout=dropout)
        att_tt = MultiHeadAttention(nb_head, d_model, dropout=dropout)

    if SE and TE:
        encode_temporal_position = TemporalPositionalEncoding(d_model, dropout, max_len, en_lookup_index)
        decode_temporal_position = TemporalPositionalEncoding(d_model, dropout, num_for_predict)
        spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout, GCN(norm_Adj_matrix, d_model, d_model), smooth_layer_num=smooth_layer_num)
        
        encoder_embedding = SpatialTemporalEmbedding(src_dense, encode_temporal_position, spatial_position)
        decoder_embedding = SpatialTemporalEmbedding(trg_dense, decode_temporal_position, spatial_position)
        
    elif SE and (not TE):
        spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout, GCN(norm_Adj_matrix, d_model, d_model), smooth_layer_num=smooth_layer_num)
        encoder_embedding = SpatialEmbedding(src_dense, spatial_position)
        decoder_embedding = SpatialEmbedding(trg_dense, spatial_position)
    elif (not SE) and (TE):
        encode_temporal_position = TemporalPositionalEncoding(d_model, dropout, max_len, en_lookup_index)
        decode_temporal_position = TemporalPositionalEncoding(d_model, dropout, num_for_predict)
        encoder_embedding = nn.Sequential(src_dense, c(encode_temporal_position))
        decoder_embedding = nn.Sequential(trg_dense, c(decode_temporal_position))
    else:
        encoder_embedding = nn.Sequential(src_dense)
        decoder_embedding = nn.Sequential(trg_dense)

    encoderLayer = EncoderLayer(d_model, attn_ss, c(position_wise_gcn), dropout, residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)
    encoder = Encoder(encoderLayer, num_layers)

    decoderLayer = DecoderLayer(d_model, att_tt, attn_st, c(position_wise_gcn), dropout, residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)
    decoder = Decoder(decoderLayer, num_layers)

    generator = nn.Linear(d_model, decoder_output_size)

    model = EncoderDecoder(encoder,
                           decoder,
                           encoder_embedding,
                           decoder_embedding,
                           generator,
                           DEVICE)
    # Parameter initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model