import torch.nn as nn
import torch
import torch.nn.functional as F
import math



class DCM(nn.Module):
    def __init__(self, hidden_size, inner_size=None, num_patch=25, drop_prob=0.):
        super(DCM, self).__init__()

        self.hidden_size = hidden_size
        self.inner_size = inner_size if inner_size is not None else hidden_size//8
        self.num_patch = num_patch

        dim_per_head = inner_size
        self.num_heads = 1
        inner_dim = self.inner_size * self.num_heads
        self.to_qkv = nn.Sequential(
            nn.Linear(self.hidden_size, inner_dim * 3, bias=False),
            )
        self.dropout = nn.Dropout(drop_prob)




    def compute_distances(self, query_a, key_a, value_a, query_b, key_b, value_b):
        value_a = value_a.unsqueeze(0)
        value_b = value_b.unsqueeze(1)

        n_way = value_a.size(1)
        n_query = value_b.size(0)
        s_patch = value_a.size(3)
        n_shot = s_patch // 25


        att_scores = torch.matmul(query_b.unsqueeze(1), key_a.unsqueeze(0).transpose(-1, -2).contiguous())

        att_probs = nn.Softmax(dim=-1)(att_scores / math.sqrt(self.inner_size))

        att_probs = self.dropout(att_probs)

        reconstructed_features_b = torch.matmul(att_probs, value_a)


        att_scores = torch.matmul(query_a.unsqueeze(0), key_b.unsqueeze(1).transpose(-1, -2).contiguous())
        att_probs = nn.Softmax(dim=-1)(att_scores / math.sqrt(self.inner_size))

        att_probs = self.dropout(att_probs)


        reconstructed_features_a = torch.matmul(att_probs, value_b)




        assert reconstructed_features_a.size(-1) == self.inner_size
        assert reconstructed_features_b.size(-1) == self.inner_size

        assert value_a.size(-1) == self.inner_size
        assert value_b.size(-1) == self.inner_size


        sq_similarity = -torch.sum((value_a.view(value_a.size(0), value_a.size(1),-1) - reconstructed_features_a.view(reconstructed_features_a.size(0), reconstructed_features_a.size(1), -1)) ** 2, dim=-1)

        qs_similarity = -torch.sum((value_b.view(value_b.size(0), value_b.size(1),-1) - reconstructed_features_b.view(reconstructed_features_b.size(0), reconstructed_features_b.size(1), -1)) ** 2, dim=-1)

        return sq_similarity, qs_similarity


    def forward(self, features_a, features_b):

        features_a = features_a.view(features_a.size(0), features_a.size(1), -1).permute(0, 2, 1).contiguous()

        b_a, l_a, d_a = features_a.shape


        qkv_a = self.to_qkv(features_a)

        qkv_a = qkv_a.view(b_a, l_a, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()

        query_a, key_a, value_a = qkv_a.chunk(3)

        query_a, key_a, value_a = query_a.squeeze(0), key_a.squeeze(0), value_a.squeeze(0)





        features_b = features_b.view(features_b.size(0), features_b.size(1), -1).permute(0, 2, 1).contiguous()

        b_b, l_b, d_b = features_b.shape


        qkv_b = self.to_qkv(features_b)

        qkv_b = qkv_b.view(b_b, l_b, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()

        query_b, key_b, value_b = qkv_b.chunk(3)
        query_b, key_b, value_b = query_b.squeeze(0), key_b.squeeze(0), value_b.squeeze(0)


        distances = self.compute_distances(query_a, key_a, value_a, query_b, key_b, value_b)

        return distances



