import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.clip_model.model import load_download_clip, Transformer

class ResidualMLPs(nn.Module):
    def __init__(self, org_dim, dropout=0., num_layers=2, activation='relu'):
        super().__init__()
        self.num_layers = num_layers

        if activation == 'relu':
            self.activation_layer = nn.ReLU()
        elif activation == 'gelu':
            self.activation_layer = nn.GELU()
        else:
            pass

        self.mlps = nn.ModuleList(nn.Sequential(
            nn.Linear(org_dim, 4 * org_dim),
            self.activation_layer,
            nn.Dropout(p=dropout),
            nn.Linear(4 * org_dim, org_dim),
        ) for i in range(num_layers))

        self.lns = nn.ModuleList(nn.LayerNorm(org_dim) for i in range(num_layers))

    def forward(self, x):
        for i in range(self.num_layers):
            x = x + self.mlps[i](self.lns[i](x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0., max_len=64):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe / (d_model ** 0.5)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class BitwiseHashing(nn.Module):
    def __init__(self, org_dim, k_bits=32):
        super().__init__()
        self.k = k_bits
        self.fc_list = nn.ModuleList(nn.Linear(org_dim, 1) for _ in range(k_bits))

    def forward(self, x):
        batch_size = x.size(1)
        hashed_outputs = []

        for i in range(self.k):
            hashed_l = [self.fc_list[i](x[l, :, :]) for l in range(x.size(0))]
            hashed_l = torch.stack(hashed_l, dim=0)
            
            hashed_output = torch.mean(hashed_l, dim=0)
            hashed_outputs.append(hashed_output)

        x = torch.cat(hashed_outputs, dim=1)
        return torch.tanh(x)

class GlobalConceptLearning(nn.Module):
    def __init__(self, k_concept, org_dim, dropout=0., activation='relu', res_mlp_layers=0):
        super().__init__()

        if res_mlp_layers != 0:
            self.mlp = ResidualMLPs(org_dim=org_dim, dropout=dropout, num_layers=res_mlp_layers, activation=activation)
        else:
            self.mlp = nn.Identity()

        self.common_concept_embedding = nn.Linear(org_dim, k_concept, bias=False)

    def forward(self, x):
        x = self.mlp(x)
        return x, torch.tanh(self.common_concept_embedding(x))

class LocalConceptTransforming(nn.Module):
    def __init__(self, clip_embed_dim, k_bits, transformer_layers, dropout):
        super().__init__()
        self.position = PositionalEncoding(clip_embed_dim, dropout=dropout)
        self.transformer = Transformer(
            width=clip_embed_dim,
            layers=transformer_layers,
            heads=clip_embed_dim // 64,
        )
        self.hashing = BitwiseHashing(org_dim=clip_embed_dim, k_bits=k_bits)

    def forward(self, x, key_padding_mask=None):
        x, _ = self.transformer(self.position(x))
        return self.hashing(x)

class HashingModel(nn.Module):
    def __init__(self, clip_info=None, args=None):
        super().__init__()

        self.k_bits = k_bits = args.k_bits
        self.dropout = dropout = args.dropout
        self.transformer_layers = transformer_layers = args.transformer_layers
        self.activation = activation = args.activation
        self.res_mlp_layers = res_mlp_layers = args.res_mlp_layers

        clip_embed_dim = clip_info['embed_dim']

        self.gcl_i = self.gcl_t = self.gcl_l = GlobalConceptLearning(k_concept=k_bits, org_dim=clip_embed_dim, dropout=dropout,
                                                        activation=activation, res_mlp_layers=res_mlp_layers)

        self.lct_i = LocalConceptTransforming(clip_embed_dim=clip_embed_dim, k_bits=k_bits,
                                              transformer_layers=transformer_layers, dropout=0)
        self.lct_t = LocalConceptTransforming(clip_embed_dim=clip_embed_dim, k_bits=k_bits,
                                              transformer_layers=transformer_layers, dropout=0)
        self.lct_l = LocalConceptTransforming(clip_embed_dim=clip_embed_dim, k_bits=k_bits,
                                              transformer_layers=transformer_layers, dropout=0)

    def forward(self, img_tokens, txt_tokens, lab_tokens, img_cls, txt_eos, lab_loc, key_padding_mask, key_padding_mask_t):
        output_dict = {}

        gcl_i = self.gcl_i
        gcl_t = self.gcl_t
        gcl_l = self.gcl_l
        lct_i = self.lct_i
        lct_t = self.lct_t
        lct_l = self.lct_l

        res_img_cls, img_cls_hash = gcl_i(img_cls)
        res_txt_cls, txt_cls_hash = gcl_t(txt_eos)
        res_lab_cls, lab_cls_hash = gcl_l(lab_loc)

        output_dict['img_cls_hash'] = img_cls_hash
        output_dict['txt_cls_hash'] = txt_cls_hash
        output_dict['lab_cls_hash'] = lab_cls_hash

        output_dict['res_img_cls'] = F.normalize(res_img_cls, dim=-1)
        output_dict['res_txt_cls'] = F.normalize(res_txt_cls, dim=-1)
        output_dict['res_lab_cls'] = F.normalize(res_lab_cls, dim=-1)

        tokens_hash_i = lct_i(img_tokens)
        tokens_hash_t = lct_t(txt_tokens, key_padding_mask)
        tokens_hash_l = lct_l(lab_tokens, key_padding_mask_t)

        output_dict['img_tokens_hash'] = tokens_hash_i
        output_dict['txt_tokens_hash'] = tokens_hash_t
        output_dict['lab_tokens_hash'] = tokens_hash_l

        return output_dict

    def forward_1(self, img_tokens, txt_tokens, img_cls, txt_eos, key_padding_mask):
        output_dict = {}

        gcl_i = self.gcl_i
        gcl_t = self.gcl_t
        lct_i = self.lct_i
        lct_t = self.lct_t

        _, img_cls_hash = gcl_i(img_cls)
        _, txt_cls_hash = gcl_t(txt_eos)

        output_dict['img_cls_hash'] = img_cls_hash
        output_dict['txt_cls_hash'] = txt_cls_hash

        tokens_hash_i = lct_i(img_tokens)
        tokens_hash_t = lct_t(txt_tokens, key_padding_mask)

        output_dict['img_tokens_hash'] = tokens_hash_i
        output_dict['txt_tokens_hash'] = tokens_hash_t

        return output_dict

class MultiLabelModalityEnhancedAttention(nn.Module):
    def __init__(self):
        super(MultiLabelModalityEnhancedAttention, self).__init__()
            	
    def forward(self, img_cls, txt_eos, lab_loc):
    	
        query1 = lab_loc
        key1 = img_cls
        value1 = img_cls
        
        query2 = lab_loc
        key2 = txt_eos
        value2 = txt_eos

        d_k1 = key1.size(-1)
        d_k2 = key2.size(-1)

        attention_scores1 = torch.matmul(query1, key1.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k1, dtype=torch.float32))
        attention_weights1 = F.softmax(attention_scores1, dim=-1)
        attention_output1 = torch.matmul(attention_weights1, value1)

        attention_scores2 = torch.matmul(query2, key2.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k2, dtype=torch.float32))
        attention_weights2 = F.softmax(attention_scores2, dim=-1)
        attention_output2 = torch.matmul(attention_weights2, value2)
        
        lab_loc = lab_loc + attention_output1 + attention_output2
        return lab_loc

class MATH(nn.Module):
    def __init__(self, args=None):
        super(MATH, self).__init__()
        self.args = args
        self.clip, clip_info = load_download_clip(self.args.clip_path)
        self.hash = HashingModel(clip_info=clip_info, args=args)
        self.mmea = MultiLabelModalityEnhancedAttention()

    def forward(self, image, text, key_padding_mask, label_t, key_padding_mask_t):
        img_tokens, _, img_cls = self.clip.encode_image(image)
        txt_tokens, _, new_key_padding_mask, txt_eos = self.clip.encode_text(text, key_padding_mask)
        lab_tokens, _, new_key_padding_mask_t, lab_loc = self.clip.encode_text(label_t, key_padding_mask_t)

        lab_loc = self.mmea(img_cls, txt_eos, lab_loc)
        output_dict = self.hash(img_tokens, txt_tokens, lab_tokens, img_cls, txt_eos, lab_loc, new_key_padding_mask, new_key_padding_mask_t)
        return output_dict

    def forward_1(self, image, text, key_padding_mask):
        img_tokens, _, img_cls = self.clip.encode_image(image)
        txt_tokens, _, new_key_padding_mask, txt_eos = self.clip.encode_text(text, key_padding_mask)
        output_dict = self.hash.forward_1(img_tokens, txt_tokens, img_cls, txt_eos, new_key_padding_mask)
        return output_dict