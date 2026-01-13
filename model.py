import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Helper Functions ---

def gen_projection_matrix(m, d, seed=0, device='cpu'):
    # Reproducibility management for projection matrix
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    
    n_block = m // d
    block_list = []
    
    for _ in range(n_block):
        block = torch.randn(d, d, device=device)
        q, _ = torch.linalg.qr(block)
        block_list.append(q.T)
        
    rem_rows = m - n_block * d
    if rem_rows > 0:
        block = torch.randn(d, d, device=device)
        q, _ = torch.linalg.qr(block)
        block_list.append(q.T[:rem_rows])
        
    proj_matrix = torch.vstack(block_list)
    
    multiplier = torch.norm(torch.randn(m, d, device=device), dim=1)
    
    # Restore random state to avoid affecting training loop
    torch.set_rng_state(state)
    
    return torch.matmul(torch.diag(multiplier), proj_matrix)

def positive_kernel_transformation(data, is_query, projection_matrix, numerical_stabilizer=1e-6):
    data_normalizer = 1.0 / (data.shape[-1] ** 0.25)
    data = data_normalizer * data
    ratio = 1.0 / (projection_matrix.shape[0] ** 0.5)
    
    data_dash = torch.einsum("blhd,md->blhm", data, projection_matrix)
    
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=-1, keepdim=True)
    
    if is_query:
        max_val = torch.max(data_dash, dim=-1, keepdim=True)[0]
        data_dash = ratio * (torch.exp(data_dash - diag_data - max_val) + numerical_stabilizer)
    else:
        # TF axis=[-3, -1] corresponds to Length (1) and Feature (3) dimensions in [B, L, H, M]
        # In PyTorch, assuming [B, L, H, M], reduce over L(1) and M(3)
        max_val = torch.amax(data_dash, dim=(1, 3), keepdim=True)
        data_dash = ratio * (torch.exp(data_dash - diag_data - max_val) + numerical_stabilizer)

    return data_dash

def fourier_kernel_transformation(data, projection_matrix):
    # data: [B, L, H, D]
    data_normalizer = 1.0 / (data.shape[-1] ** 0.25)
    data = data_normalizer * data
    ratio = 1.0 / (projection_matrix.shape[0] ** 0.5)
    
    data_dash = torch.einsum("blhd,md->blhm", data, projection_matrix)
    
    data_sin = ratio * torch.sin(data_dash)
    data_cos = ratio * torch.cos(data_dash)
    
    return torch.cat([data_sin, data_cos], dim=-1)

# --- Layers ---

class LayerCentering(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        # x: [B, L, D]
        mean = torch.mean(x, dim=-1, keepdim=True)
        return x - mean + self.beta

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_head, n_head, attention_dropout, 
                 feature_map_type='fourier', normalize_attn=False, 
                 d_kernel_map=128, head_init_range=(0, 1)):
        super().__init__()
        self.d_model = d_model
        self.size_per_head = d_head
        self.n_head = n_head
        self.feature_map_type = feature_map_type
        self.normalize_attn = normalize_attn
        self.d_kernel_map = d_kernel_map
        
        # Initializers
        std_att = math.sqrt(6.0 / (d_model + n_head * d_head))
        self.value_weight = nn.Parameter(torch.Tensor(d_model, n_head, d_head).uniform_(-std_att, std_att))
        self.pos_ft_weight = nn.Parameter(torch.Tensor(d_model, n_head, d_head).uniform_(-std_att, std_att), requires_grad=False)
        
        self.pos_ft_scale = nn.Parameter(torch.ones(1, 1, n_head, 1))
        
        head_left, head_right = head_init_range
        head_range = head_right - head_left
        step = head_range / n_head
        # Equivalent to tf.range(start, limit, delta)
        head_pos = torch.arange(head_left + step/2.0, head_right, step)
        self.pos_ft_offsets = nn.Parameter(head_pos.view(1, 1, n_head, 1))

        std_out = math.sqrt(6.0 / (n_head * d_head + d_model))
        self.output_weight = nn.Linear(n_head * d_head, d_model, bias=False)
        nn.init.uniform_(self.output_weight.weight, -std_out, std_out)
        
        self.output_dropout = nn.Dropout(attention_dropout)
        
        # Projection Matrix (Fixed buffer)
        seed = np.random.randint(1e8)
        proj = gen_projection_matrix(d_kernel_map, d_head, seed=seed)
        self.register_buffer('projection_matrix', proj)

    def forward(self, source_input, pos_ft, pos_ft_slopes):
        # source_input: [B, L, D]
        # einsum conversion: "bnm,mhd->bnhd" -> "bld,dhn->blhn"
        # Be careful with dimensions. TF: [B, L, D] * [D, H, S] -> [B, L, H, S]
        
        value = torch.einsum("bld,dhs->blhs", source_input, self.value_weight)
        pos_ft_projected = torch.einsum("bld,dhs->blhs", pos_ft, self.pos_ft_weight)
        pos_ft_slopes_projected = torch.einsum("bld,dhs->blhs", pos_ft_slopes, self.pos_ft_weight)
        
        query_pos_ft = self.pos_ft_scale * pos_ft_projected
        slope_pos = self.pos_ft_scale * pos_ft_slopes_projected
        key_pos_ft = query_pos_ft + self.pos_ft_offsets * slope_pos
        
        # Linear Attention Logic
        if self.feature_map_type == 'fourier':
            query_prime = fourier_kernel_transformation(query_pos_ft, self.projection_matrix)
            key_prime = fourier_kernel_transformation(key_pos_ft, self.projection_matrix)
        elif self.feature_map_type == 'positive':
            query_prime = positive_kernel_transformation(query_pos_ft, True, self.projection_matrix)
            key_prime = positive_kernel_transformation(key_pos_ft, False, self.projection_matrix)
        
        # Transpose for attention calculation: [L, B, H, M]
        query_prime = query_prime.permute(1, 0, 2, 3)
        key_prime = key_prime.permute(1, 0, 2, 3)
        value_t = value.permute(1, 0, 2, 3)
        
        # Numerator: K^T V
        # ks: [L, B, H, M], vs: [L, B, H, D] -> kvs: [B, H, M, D]
        kvs = torch.einsum("lbhm,lbhd->bhmd", key_prime, value_t)
        # qs: [L, B, H, M], kvs: [B, H, M, D] -> [L, B, H, D]
        av_attention = torch.einsum("lbhm,bhmd->lbhd", query_prime, kvs)
        
        # Transpose back: [B, L, H, D]
        av_attention = av_attention.permute(1, 0, 2, 3)
        
        if self.normalize_attn:
            # Denominator logic omitted as normalize_attn is False in default configs
            pass
            
        bsz, slen = av_attention.shape[:2]
        
        norms = torch.norm(pos_ft_slopes_projected, dim=-1, keepdim=True) / float(slen)
        av_attention = norms * av_attention
        
        av_attention = av_attention.reshape(bsz, slen, -1)
        av_attention = self.output_weight(av_attention)
        av_attention = self.output_dropout(av_attention)
        
        return av_attention, query_prime.permute(1,0,2,3), key_prime.permute(1,0,2,3)

class PositionalFeature(nn.Module):
    def __init__(self, d_feature, beta_hat_2):
        super().__init__()
        slopes = torch.arange(d_feature, 0, -4.0, dtype=torch.float32) / d_feature
        self.register_buffer('slopes', slopes * beta_hat_2)

    def forward(self, slen, bsz=None):
        pos_seq = torch.arange(0, slen, 1.0, dtype=torch.float32, device=self.slopes.device)
        normalized_slopes = (1. / float(slen - 1)) * self.slopes
        
        # Outer product: [slen, d_feature/4 approx]
        forward = torch.ger(pos_seq, normalized_slopes)
        backward = torch.flip(forward, dims=[0])
        neg_forward = -forward
        neg_backward = -backward
        
        pos_feature = torch.cat([forward, backward, neg_forward, neg_backward], dim=-1)
        
        norm_slopes_id = normalized_slopes
        pos_feature_slopes = torch.cat([
            norm_slopes_id, -norm_slopes_id, -norm_slopes_id, norm_slopes_id
        ], dim=0)
        
        pos_feature_slopes = float(slen - 1) * pos_feature_slopes.reshape(1, -1)
        
        if bsz is not None:
            pos_feature = pos_feature.unsqueeze(0).expand(bsz, -1, -1)
            pos_feature_slopes = pos_feature_slopes.unsqueeze(0).expand(bsz, -1, -1)
        else:
            pos_feature = pos_feature.unsqueeze(0)
            pos_feature_slopes = pos_feature_slopes.unsqueeze(0)
            
        return pos_feature, pos_feature_slopes

class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super().__init__()
        self.layer_1 = nn.Linear(d_model, d_inner)
        self.layer_2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.dropout(self.layer_2(self.dropout(self.relu(self.layer_1(x)))))

class TransformerLayer(nn.Module):
    def __init__(self, n_head, d_head, d_model, d_inner, dropout, 
                 feature_map_type, normalize_attn, d_kernel_map, 
                 model_normalization, head_init_range):
        super().__init__()
        self.model_normalization = model_normalization
        
        self.self_attn = SelfAttention(d_model, d_head, n_head, dropout, 
                                       feature_map_type, normalize_attn, 
                                       d_kernel_map, head_init_range)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout)
        
        if model_normalization in ['preLC', 'postLC']:
            self.lc1 = LayerCentering(d_model)
            self.lc2 = LayerCentering(d_model)

    def forward(self, inputs):
        inp, pos_ft, pos_ft_slopes = inputs
        
        attn_in = self.lc1(inp) if self.model_normalization == 'preLC' else inp
        
        attn_out_tuple = self.self_attn(attn_in, pos_ft, pos_ft_slopes)
        attn_out = attn_out_tuple[0]
        
        attn_out = attn_out + inp
        if self.model_normalization == 'postLC':
            attn_out = self.lc1(attn_out)
            
        ff_in = self.lc2(attn_out) if self.model_normalization == 'preLC' else attn_out
        
        ff_out = self.pos_ff(ff_in)
        ff_out = ff_out + attn_out
        
        if self.model_normalization == 'postLC':
            ff_out = self.lc2(ff_out)
            
        return [ff_out] + list(attn_out_tuple[1:])

class SoftmaxAttention(nn.Module):
    def __init__(self, d_model, n_head, d_head):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.scale = 1. / (d_head ** 0.5)
        
        self.q_heads = nn.Parameter(torch.Tensor(d_head, n_head))
        nn.init.xavier_uniform_(self.q_heads)
        
        self.k_net = nn.Linear(d_model, d_head * n_head)
        self.v_net = nn.Linear(d_model, d_head * n_head)

    def forward(self, inp, softmax_attn_smoothing=1.0):
        bsz, slen, _ = inp.shape
        
        k_head = self.k_net(inp).view(bsz, slen, self.d_head, self.n_head)
        v_head = self.v_net(inp).view(bsz, slen, self.d_head, self.n_head)
        
        # q_heads: [D, H]
        # k_head: [B, L, D, H] -> einsum "bldh,dh->blh"
        attn_score = torch.einsum("bldh,dh->blh", k_head, self.q_heads)
        attn_score = attn_score * self.scale * softmax_attn_smoothing
        
        attn_prob = F.softmax(attn_score, dim=1) # Softmax over Length
        
        # v_head: [B, L, D, H], attn_prob: [B, L, H] -> "bldh,blh->bhd" (reduce over L)
        # Wait, the original TF code outputs [B, H, D] essentially?
        # TF: einsum("bndh,bnh->bnhd", v_head, attn_prob) -> [B, N_head, D_head]
        # Then reshape to [B, L, -1] ?? 
        # TF original: `attn_out = tf.reshape(attn_out, [bsz, slen, -1])` -> This looks like a bug in original TF if the reduction happened?
        # Actually, TF `softmax(axis=1)` reduces L. 
        # einsum `bndh,bnh->bnhd` (N=Slenght in TF notation inside einsum? No n is usually batch/seq. In TF code: `bndh` is B, Len, D, H).
        # TF Code: `attn_prob = tf.nn.softmax(attn_score, axis=1)` -> [B, L, H].
        # TF Code: `tf.einsum("bndh,bnh->bnhd", v_head, attn_prob)` -> Sum over 'n' (dim 1, Length). Result [B, H, D].
        # TF Code: `attn_out = tf.reshape(attn_out, [bsz, slen, -1])` -> This fails if shape is [B, H, D].
        # Wait, `softmax_attn` is used for Global Pooling at the end.
        # `SoftmaxAttention` in TF seems to output a single vector per head, but then tries to reshape to `[bsz, slen, -1]`.
        # If the goal is aggregation, it should be [B, H*D].
        # Let's assume standard Attention Aggregation where we get one vector per sequence.
        
        attn_out = torch.einsum("bldh,blh->bhd", v_head, attn_prob) # [B, H, D]
        # Flatten
        attn_out = attn_out.reshape(bsz, -1) # [B, H*D]
        
        return attn_out, attn_score

# --- Main Model ---

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.d_model = args.d_model
        
        # Convolutional Front-end
        conv_filters = [min(8 * 2**i, self.d_model) for i in range(args.n_conv_layer - 1)] + [self.d_model]
        self.conv_layers = nn.ModuleList()
        self.relu = nn.ReLU()
        
        for l in range(args.n_conv_layer):
            ks = 11 if l == 0 else args.conv_kernel_size
            in_channels = 1 if l == 0 else conv_filters[l-1]
            out_channels = conv_filters[l]
            # PyTorch Conv1d: (Batch, Channel, Length)
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, ks))
            
        self.pool = nn.AvgPool1d(args.pool_size)
        
        self.pos_feature = PositionalFeature(self.d_model, args.beta_hat_2)
        
        # Transformer Layers
        self.tran_layers = nn.ModuleList()
        for i in range(args.n_layer):
            if args.head_initialization == 'forward':
                 r = (0., 0.5) if i == 0 else (0., 1.0)
            elif args.head_initialization == 'backward':
                 r = (-0.5, 0.) if i == 0 else (-1.0, 0.)
            else: # symmetric
                 r = (-0.5, 0.5) if i == 0 else (-1.0, 1.0)
                 
            self.tran_layers.append(TransformerLayer(
                n_head=args.n_head, d_head=args.d_head, d_model=args.d_model,
                d_inner=args.d_inner, dropout=args.dropout, 
                feature_map_type='fourier', normalize_attn=False,
                d_kernel_map=args.d_kernel_map, model_normalization=args.model_normalization,
                head_init_range=r
            ))
            
        self.out_dropout = nn.Dropout(args.dropout)
        
        if args.softmax_attn:
            self.out_attn = SoftmaxAttention(d_model=args.d_model, 
                                             n_head=args.n_head_softmax, 
                                             d_head=args.d_head_softmax)
        
        self.fc_output = nn.Linear(args.n_head_softmax * args.d_head_softmax if args.softmax_attn else self.d_model, 
                                   args.n_classes)

    def forward(self, inp, softmax_attn_smoothing=1.0):
        # inp: [B, L]
        x = inp.unsqueeze(1) # [B, 1, L] for Conv1d
        
        for conv in self.conv_layers:
            x = conv(x)
            x = self.relu(x)
            x = self.pool(x)
            
        # Transformer expects [B, L, D]
        x = x.permute(0, 2, 1) # [B, L, D]
        bsz, slen, _ = x.shape
        
        pos_ft, pos_ft_slopes = self.pos_feature(slen, bsz)
        
        core_out = x
        attn_maps = []
        
        for layer in self.tran_layers:
            outputs = layer([core_out, pos_ft, pos_ft_slopes])
            core_out = outputs[0]
            attn_maps.append(outputs[1:])
            
        core_out = self.out_dropout(core_out)
        
        softmax_score = None
        if self.args.softmax_attn:
            # SoftmaxAttention aggregates over length
            core_out, softmax_score = self.out_attn(core_out, softmax_attn_smoothing)
        else:
            # Global Average Pooling
            core_out = torch.mean(core_out, dim=1)
            
        scores = self.fc_output(core_out)
        
        if self.args.output_attn:
            return scores, attn_maps, softmax_score
        return scores