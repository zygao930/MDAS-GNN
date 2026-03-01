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
    return torch.from_numpy(subsequent_mask) == 0  

class spatialGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
        super(spatialGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels)) 
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))

class GCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
        super(GCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x))) 

class FeatureSpecificSpatialAttention(nn.Module):
    def __init__(self, d_model, num_features=3, dropout=0.0):
        super(FeatureSpecificSpatialAttention, self).__init__()
        self.d_model = d_model
        self.num_features = num_features
        self.dropout = nn.Dropout(p=dropout)
        self.feature_transforms = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(num_features)
        ])

    def forward(self, x):
        batch_size, num_vertices, num_timesteps, d_model = x.shape
        attended_features = []
        
        for f in range(self.num_features):
            x_reshaped = x.permute(0, 2, 1, 3).reshape(-1, num_vertices, d_model)
            scores = torch.matmul(x_reshaped, x_reshaped.transpose(-2, -1)) / math.sqrt(d_model)
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
 
            attended = torch.matmul(attention_weights, x_reshaped)
            attended = self.feature_transforms[f](attended)
            attended = attended.reshape(batch_size, num_timesteps, num_vertices, d_model).permute(0, 2, 1, 3)
            attended_features.append(attended)

        return torch.stack(attended_features, dim=0).mean(dim=0)

class Spatial_Attention_layer(nn.Module):
    def __init__(self, dropout=.0):
        super(Spatial_Attention_layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels)) 
        score = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(in_channels)  
        score = self.dropout(F.softmax(score, dim=-1)) 
        return score.reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices))

class spatialAttentionGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, dropout=.0):
        super(spatialAttentionGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.SAt = Spatial_Attention_layer(dropout=dropout)

    def forward(self, x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        spatial_attention = self.SAt(x) 
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels)) 
        spatial_attention = spatial_attention.reshape((-1, num_of_vertices, num_of_vertices)) 
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix.mul(spatial_attention), x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))
 
class spatialAttentionScaledGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, dropout=.0):
        super(spatialAttentionScaledGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.SAt = Spatial_Attention_layer(dropout=dropout)

    def forward(self, x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        spatial_attention = self.SAt(x) / math.sqrt(in_channels) 
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))
        spatial_attention = spatial_attention.reshape((-1, num_of_vertices, num_of_vertices)) 
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix.mul(spatial_attention), x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))

class FeatureSpecificSpatialAttentionGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, dropout=.0):
        super(FeatureSpecificSpatialAttentionGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.feature_spatial_attn = FeatureSpecificSpatialAttention(in_channels, dropout=dropout)

    def forward(self, x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x_attended = self.feature_spatial_attn(x)
        x_attended = x_attended.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels)) 
        output = F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x_attended)))
        
        return output.reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2)

class FeatureSpecificSpatialAttentionScaledGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, dropout=.0):
        super(FeatureSpecificSpatialAttentionScaledGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.feature_spatial_attn = FeatureSpecificSpatialAttention(in_channels, dropout=dropout)

    def forward(self, x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x_attended = self.feature_spatial_attn(x) / math.sqrt(in_channels)

        x_attended = x_attended.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels)) 
        output = F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x_attended)))
        
        return output.reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2)

class SpatialPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_of_vertices, dropout, gcn=None, smooth_layer_num=0):
        super(SpatialPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = torch.nn.Embedding(num_of_vertices, d_model)
        self.gcn_smooth_layers = None
        if (gcn is not None) and (smooth_layer_num > 0):
            self.gcn_smooth_layers = nn.ModuleList([gcn for _ in range(smooth_layer_num)])

    def forward(self, x):
        batch, num_of_vertices, timestamps, _ = x.shape
        x_indexs = torch.LongTensor(torch.arange(num_of_vertices)).to(x.device) 
        embed = self.embedding(x_indexs).unsqueeze(0)  
        if self.gcn_smooth_layers is not None:
            for _, l in enumerate(self.gcn_smooth_layers):
                embed = l(embed) 
        x = x + embed.unsqueeze(2)  
        return self.dropout(x)

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len, lookup_index=None):
        super(TemporalPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.lookup_index = lookup_index
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0).unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.lookup_index is not None:
            x = x + self.pe[:, :, self.lookup_index, :]  
        else:
            x = x + self.pe[:, :, :x.size(2), :]
        return self.dropout(x.detach())

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout, residual_connection, use_LayerNorm):
        super(SublayerConnection, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.dropout = nn.Dropout(dropout)
        if self.use_LayerNorm:
            self.norm = nn.LayerNorm(size)

    def forward(self, x, sublayer):
        if self.residual_connection and self.use_LayerNorm:
            return x + self.dropout(sublayer(self.norm(x)))
        if self.residual_connection and (not self.use_LayerNorm):
            return x + self.dropout(sublayer(x))
        if (not self.residual_connection) and self.use_LayerNorm:
            return self.dropout(sublayer(self.norm(x)))

class PositionWiseGCNFeedForward(nn.Module):
    def __init__(self, gcn, dropout=.0):
        super(PositionWiseGCNFeedForward, self).__init__()
        self.gcn = gcn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(F.relu(self.gcn(x)))

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  

    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9) 
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn 

class MultiHeadAttention(nn.Module):
    def __init__(self, nb_head, d_model, dropout=.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1) 

        nbatches = query.size(0)
        N = query.size(1)

        query, key, value = [l(x).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3) for l, x in
                             zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(2, 3).contiguous()  
        x = x.view(nbatches, N, -1, self.h * self.d_k) 
        return self.linears[-1](x)

class MultiHeadAttentionAwareTemporalContex_qc_kc(nn.Module): 
    def __init__(self, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size=3, dropout=.0):
        super(MultiHeadAttentionAwareTemporalContex_qc_kc, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)  
        self.padding = kernel_size - 1
        self.conv1Ds_aware_temporal_context = clones(nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)), 2) 
        self.dropout = nn.Dropout(p=dropout)
        self.w_length = num_of_weeks * points_per_hour
        self.d_length = num_of_days * points_per_hour
        self.h_length = num_of_hours * points_per_hour

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1) 

        nbatches = query.size(0)
        N = query.size(1)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []
            if self.w_length > 0:
                query_w, key_w = [l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, :self.w_length, :], key[:, :, :self.w_length, :]))]
                query_list.append(query_w)
                key_list.append(key_w)

            if self.d_length > 0:
                query_d, key_d = [l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, self.w_length:self.w_length+self.d_length, :], key[:, :, self.w_length:self.w_length+self.d_length, :]))]
                query_list.append(query_d)
                key_list.append(key_d)

            if self.h_length > 0:
                query_h, key_h = [l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :], key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :]))]
                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):
            query, key = [l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query, key))]

        elif (not query_multi_segment) and (key_multi_segment):
            query = self.conv1Ds_aware_temporal_context[0](query.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
            key_list = []

            if self.w_length > 0:
                key_w = self.conv1Ds_aware_temporal_context[1](key[:, :, :self.w_length, :].permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_w)

            if self.d_length > 0:
                key_d = self.conv1Ds_aware_temporal_context[1](key[:, :, self.w_length:self.w_length + self.d_length, :].permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_d)

            if self.h_length > 0:
                key_h = self.conv1Ds_aware_temporal_context[1](key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(2, 3).contiguous()  
        x = x.view(nbatches, N, -1, self.h * self.d_k)  
        return self.linears[-1](x)

class MultiHeadAttentionAwareTemporalContex_q1d_k1d(nn.Module): 
    def __init__(self, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size=3, dropout=.0):

        super(MultiHeadAttentionAwareTemporalContex_q1d_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2) 
        self.padding = (kernel_size - 1)//2

        self.conv1Ds_aware_temporal_context = clones(
            nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)),
            2) 

        self.dropout = nn.Dropout(p=dropout)
        self.w_length = num_of_weeks * points_per_hour
        self.d_length = num_of_days * points_per_hour
        self.h_length = num_of_hours * points_per_hour

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1) 
        nbatches = query.size(0)
        N = query.size(1)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []
            if self.w_length > 0:
                query_w, key_w = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, :self.w_length, :], key[:, :, :self.w_length, :]))]
                query_list.append(query_w)
                key_list.append(key_w)

            if self.d_length > 0:
                query_d, key_d = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, self.w_length:self.w_length+self.d_length, :], key[:, :, self.w_length:self.w_length+self.d_length, :]))]
                query_list.append(query_d)
                key_list.append(key_d)

            if self.h_length > 0:
                query_h, key_h = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :], key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :]))]
                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):
            query, key = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query, key))]

        elif (not query_multi_segment) and (key_multi_segment):
            query = self.conv1Ds_aware_temporal_context[0](query.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

            key_list = []

            if self.w_length > 0:
                key_w = self.conv1Ds_aware_temporal_context[1](key[:, :, :self.w_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_w)

            if self.d_length > 0:
                key_d = self.conv1Ds_aware_temporal_context[1](key[:, :, self.w_length:self.w_length + self.d_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_d)

            if self.h_length > 0:
                key_h = self.conv1Ds_aware_temporal_context[1](key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(2, 3).contiguous() 
        x = x.view(nbatches, N, -1, self.h * self.d_k)  
        return self.linears[-1](x)

class MultiHeadAttentionAwareTemporalContex_qc_k1d(nn.Module): 
    def __init__(self, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size=3, dropout=.0):
        super(MultiHeadAttentionAwareTemporalContex_qc_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2) 
        self.causal_padding = kernel_size - 1
        self.padding_1D = (kernel_size - 1)//2
        self.query_conv1Ds_aware_temporal_context = nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.causal_padding))
        self.key_conv1Ds_aware_temporal_context = nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding_1D))
        self.dropout = nn.Dropout(p=dropout)
        self.w_length = num_of_weeks * points_per_hour
        self.d_length = num_of_days * points_per_hour
        self.h_length = num_of_hours * points_per_hour

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  
        nbatches = query.size(0)
        N = query.size(1)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []
            if self.w_length > 0:
                query_w = self.query_conv1Ds_aware_temporal_context(query[:, :, :self.w_length, :].permute(0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_w = self.key_conv1Ds_aware_temporal_context(key[:, :, :self.w_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                query_list.append(query_w)
                key_list.append(key_w)

            if self.d_length > 0:
                query_d = self.query_conv1Ds_aware_temporal_context(query[:, :, self.w_length:self.w_length+self.d_length, :].permute(0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_d = self.key_conv1Ds_aware_temporal_context(key[:, :, self.w_length:self.w_length+self.d_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                query_list.append(query_d)
                key_list.append(key_d)

            if self.h_length > 0:
                query_h = self.query_conv1Ds_aware_temporal_context(query[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_h = self.key_conv1Ds_aware_temporal_context(key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):

            query = self.query_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
            key = self.key_conv1Ds_aware_temporal_context(key.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

        elif (not query_multi_segment) and (key_multi_segment):

            query = self.query_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

            key_list = []

            if self.w_length > 0:
                key_w = self.key_conv1Ds_aware_temporal_context(key[:, :, :self.w_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_w)

            if self.d_length > 0:
                key_d = self.key_conv1Ds_aware_temporal_context(key[:, :, self.w_length:self.w_length + self.d_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_d)

            if self.h_length > 0:
                key_h = self.key_conv1Ds_aware_temporal_context(key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2)).contiguous().view(
                    nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(2, 3).contiguous()  
        x = x.view(nbatches, N, -1, self.h * self.d_k) 
        return self.linears[-1](x)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_dense, trg_dense, generator, DEVICE):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_dense
        self.trg_embed = trg_dense
        self.prediction_generator = generator
        self.to(DEVICE)

    def forward(self, src, trg):
        encoder_output = self.encode(src) 
        return self.decode(trg, encoder_output)

    def encode(self, src):
        h = self.src_embed(src)
        return self.encoder(h)

    def decode(self, trg, encoder_output):
        return self.prediction_generator(self.decoder(self.trg_embed(trg), encoder_output))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, gcn, dropout, residual_connection=True, use_LayerNorm=True):
        super(EncoderLayer, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.self_attn = self_attn
        self.feed_forward_gcn = gcn
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 2)
        self.size = size

    def forward(self, x):
        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, query_multi_segment=True, key_multi_segment=True))
            return self.sublayer[1](x, self.feed_forward_gcn)
        else:
            x = self.self_attn(x, x, x, query_multi_segment=True, key_multi_segment=True)
            return self.feed_forward_gcn(x)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, gcn, dropout, residual_connection=True, use_LayerNorm=True):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward_gcn = gcn
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 3)

    def forward(self, x, memory):
        m = memory
        tgt_mask = subsequent_mask(x.size(-2)).to(m.device)  # (1, T', T')
        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, query_multi_segment=False, key_multi_segment=False))  # output: (batch, N, T', d_model)
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, query_multi_segment=False, key_multi_segment=True))  # output: (batch, N, T', d_model)
            return self.sublayer[2](x, self.feed_forward_gcn)  # output:  (batch, N, T', d_model)
        else:
            x = self.self_attn(x, x, x, tgt_mask, query_multi_segment=False, key_multi_segment=False)  # output: (batch, N, T', d_model)
            x = self.src_attn(x, m, m, query_multi_segment=False, key_multi_segment=True)  # output: (batch, N, T', d_model)
            return self.feed_forward_gcn(x)  # output:  (batch, N, T', d_model)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory):
        for layer in self.layers:
            x = layer(x, memory)
        return self.norm(x)

def search_index(max_len, num_of_depend, num_for_predict,points_per_hour, units):
    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = max_len - points_per_hour * units * i
        for j in range(num_for_predict):
            end_idx = start_idx + j
            x_idx.append(end_idx)
    return x_idx

class SpatialTemporalEmbedding(nn.Module):
    def __init__(self, dense, temporal_pos, spatial_pos, landuse_embed=None):
        super(SpatialTemporalEmbedding, self).__init__()
        self.dense = dense
        self.temporal_pos = temporal_pos
        self.spatial_pos = spatial_pos
        self.landuse_embed = landuse_embed
        
    def forward(self, x):
        x = self.dense(x)
        x = self.temporal_pos(x)
        x = self.spatial_pos(x)
        if self.landuse_embed is not None:
            x = x + self.landuse_embed.unsqueeze(0).unsqueeze(2)
        return x

class SpatialEmbedding(nn.Module):
    def __init__(self, dense, spatial_pos, landuse_embed=None):
        super(SpatialEmbedding, self).__init__()
        self.dense = dense
        self.spatial_pos = spatial_pos
        self.landuse_embed = landuse_embed
        
    def forward(self, x):
        x = self.dense(x)
        x = self.spatial_pos(x)
        if self.landuse_embed is not None:
            x = x + self.landuse_embed.unsqueeze(0).unsqueeze(2)
        return x

def make_model(DEVICE, num_layers, encoder_input_size, decoder_output_size, d_model, adj_mx, nb_head, num_of_weeks,num_of_days, 
              num_of_hours, points_per_hour, num_for_predict, dropout=.0, aware_temporal_context=True,ScaledSAt=True, SE=True, 
              TE=True, kernel_size=3, smooth_layer_num=0, residual_connection=True, use_LayerNorm=True, landuse_features_path=None
              ):

    c = copy.deepcopy
    norm_Adj_matrix = torch.from_numpy(norm_Adj(adj_mx)).type(torch.FloatTensor).to(DEVICE)
    num_of_vertices = norm_Adj_matrix.shape[0]
    src_dense = nn.Linear(encoder_input_size, d_model)

    landuse_embed = None
    if landuse_features_path and SE:
        try:
            landuse_features = np.load(landuse_features_path)  
            if landuse_features.shape[0] == num_of_vertices:
                landuse_embed = torch.from_numpy(landuse_features).type(torch.FloatTensor).to(DEVICE)
                print(f"Successfully loaded land use features: {landuse_embed.shape}")
            else:
                print(f"Warning: Land use feature count ({landuse_features.shape[0]}) doesn't match vertex count ({num_of_vertices})")
        except Exception as e:
            print(f"Failed to load land use features: {e}")

    if ScaledSAt:
        position_wise_gcn = PositionWiseGCNFeedForward(FeatureSpecificSpatialAttentionScaledGCN(norm_Adj_matrix, d_model, d_model), dropout=dropout)
    else:
        position_wise_gcn = PositionWiseGCNFeedForward(FeatureSpecificSpatialAttentionGCN(norm_Adj_matrix, d_model, d_model), dropout=dropout)

    trg_dense = nn.Linear(decoder_output_size, d_model)
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
        
        encoder_embedding = SpatialTemporalEmbedding(src_dense, encode_temporal_position, spatial_position, landuse_embed)
        decoder_embedding = SpatialTemporalEmbedding(trg_dense, decode_temporal_position, spatial_position, landuse_embed)
        
    elif SE and (not TE):
        spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout, GCN(norm_Adj_matrix, d_model, d_model), smooth_layer_num=smooth_layer_num)
        encoder_embedding = SpatialEmbedding(src_dense, spatial_position, landuse_embed)
        decoder_embedding = SpatialEmbedding(trg_dense, spatial_position, landuse_embed)
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

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
