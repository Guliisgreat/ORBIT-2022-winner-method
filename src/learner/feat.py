import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import collections
from einops import rearrange

from src.official_orbit.models.classifiers import HeadClassifier


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    """
        Multi-Head Attention module

        Copy from https://github.com/Sha-Lab/FEAT/blob/47bdc7c1672e00b027c67469d0291e7502918950/model/models/feat.py
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class FEAT(HeadClassifier):
    """
        Creates instance of FEAT.

        FEAT is a metric-based few shot learning method, where a transformer encoder block, as the embedding adaptation
        module, is applied on prototypes to enable learn more discriminative representations of the specific episode.

        Details in Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions, CVPR 2020
        <https://openaccess.thecvf.com/content_CVPR_2020/papers/Ye_Few-Shot_Learning_via_Embedding_Adaptation_With_Set-to-Set_Functions_CVPR_2020_paper.pdf>

    """
    def __init__(self, hidden_dim=1280, temperature=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.param_dict = {}
        self.hidden_dim = hidden_dim
        self.multi_head_self_attention = MultiHeadAttention(1, self.hidden_dim, self.hidden_dim, self.hidden_dim,
                                                            dropout=0.5)

        self.original_class_rep_dict = None
        self.adapted_class_rep_dict = None

    def apply_embedding_adaptation(self, class_rep_dict: collections.OrderedDict):
        prototypes = torch.cat(list(class_rep_dict.values()))
        prototypes = rearrange(prototypes, "n h -> 1 n h")
        prototypes = self.multi_head_self_attention(prototypes, prototypes, prototypes)
        class_rep_dict = collections.OrderedDict(
            {idx: n for idx, n in enumerate(rearrange(prototypes, "1 n h -> n 1 h"))})
        return class_rep_dict

    def predict(self, target_features):
        logits = F.linear(target_features, self.param_dict['weight'], self.param_dict['bias'])
        return logits / self.temperature

    def configure(self, context_features, context_labels, ops_counter=None):
        assert context_features.size(0) == context_labels.size(0), "context features and labels are different sizes!"

        self.original_class_rep_dict = self._build_class_reps(context_features, context_labels, ops_counter)  # List[(1, hidden_dim)]
        self.adapted_class_rep_dict = self.apply_embedding_adaptation(self.original_class_rep_dict)

        class_weight = []
        class_bias = []

        label_set = list(self.adapted_class_rep_dict.keys())
        label_set.sort()
        num_classes = len(label_set)

        for class_num in label_set:
            # equation 8 from the prototypical networks paper
            nu = self.adapted_class_rep_dict[class_num]
            class_weight.append(2 * nu)
            class_bias.append((-torch.matmul(nu, nu.t()))[None, None])

        self.param_dict['weight'] = torch.cat(class_weight, dim=0)
        self.param_dict['bias'] = torch.reshape(torch.cat(class_bias, dim=1), [num_classes, ])
