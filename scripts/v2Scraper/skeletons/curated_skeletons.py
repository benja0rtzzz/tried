ENTRIES = [
    {
        "name": "gelu_tanh_approx",
        "op_category": "elementwise_chain",
        "input_shapes": [[4096, 1024]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    # tanh approximation of GELU used in GPT-2 / PaLM
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
""",
    },
    {
        "name": "silu_gated",
        "op_category": "elementwise_chain",
        "input_shapes": [[4, 512, 2048], [4, 512, 2048]],
        "input_dtypes": ["float16", "float16"],
        "code": """
def op(x, gate):
    # gated SiLU / SwiGLU pattern used in LLaMA FFN
    return x * F.silu(gate)
""",
    },
    {
        "name": "hard_swish",
        "op_category": "elementwise_chain",
        "input_shapes": [[2048, 512]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    return x * F.relu6(x + 3.0) / 6.0
""",
    },
    {
        "name": "rms_per_row",
        "op_category": "reduction",
        "input_shapes": [[64, 512]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    return torch.sqrt((x * x).mean(dim=-1))
""",
    },
    {
        "name": "manual_softmax",
        "op_category": "elementwise_chain",
        "input_shapes": [[32, 128]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    x_max = x.max(dim=-1, keepdim=True).values
    e = torch.exp(x - x_max)
    return e / e.sum(dim=-1, keepdim=True)
""",
    },
    {
        "name": "layer_norm_manual",
        "op_category": "normalization",
        "input_shapes": [[8, 256, 512], [512], [512]],
        "input_dtypes": ["float32", "float32", "float32"],
        "code": """
def op(x, weight, bias):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return weight * (x - mean) / torch.sqrt(var + 1e-5) + bias
""",
    },
    {
        "name": "linear_gelu",
        "op_category": "matmul",
        "input_shapes": [[32, 256], [512, 256], [512]],
        "input_dtypes": ["float32", "float32", "float32"],
        "code": """
def op(x, w, b):
    return F.gelu(F.linear(x, w, b))
""",
    },
    {
        "name": "adaln_scale_shift",
        "op_category": "elementwise_chain",
        "input_shapes": [[8, 256, 512], [8, 256, 512], [512], [512]],
        "input_dtypes": ["float32", "float32", "float32", "float32"],
        "code": """
def op(x, residual, scale, shift):
    # AdaLN modulation pattern from DiT / diffusion transformers
    return (x + residual) * (1.0 + scale) + shift
""",
    },
    {
        "name": "exp_clamp_log",
        "op_category": "elementwise_chain",
        "input_shapes": [[1048576]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    return torch.log(torch.clamp(torch.exp(x), min=1e-6, max=1e6))
""",
    },
    {
        "name": "mish",
        "op_category": "activation",
        "input_shapes": [[2048, 512]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    return x * torch.tanh(F.softplus(x))
""",
    },
    {
        "name": "variance_per_row",
        "op_category": "reduction",
        "input_shapes": [[64, 256]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    mean = x.mean(dim=-1, keepdim=True)
    return ((x - mean) ** 2).mean(dim=-1)
""",
    },
    {
        "name": "mse_loss_manual",
        "op_category": "loss",
        "input_shapes": [[32, 256], [32, 256]],
        "input_dtypes": ["float32", "float32"],
        "code": """
def op(pred, target):
    return ((pred - target) ** 2).mean()
""",
    },
    {
        "name": "bmm_relu",
        "op_category": "matmul",
        "input_shapes": [[16, 64, 128], [16, 128, 256]],
        "input_dtypes": ["float32", "float32"],
        "code": """
def op(a, b):
    return F.relu(torch.bmm(a, b))
""",
    },
    {
        "name": "embedding_lookup",
        "op_category": "embedding",
        "input_shapes": [[50000, 256], [32, 128]],
        "input_dtypes": ["float32", "int64"],
        "code": """
def op(weight, indices):
    return F.embedding(indices, weight)
""",
    },
    {
        "name": "scaled_dot_product_attn",
        "op_category": "fused_attention",
        "input_shapes": [[4, 8, 64, 64], [4, 8, 64, 64], [4, 8, 64, 64]],
        "input_dtypes": ["float16", "float16", "float16"],
        "code": """
def op(q, k, v):
    scale = q.shape[-1] ** -0.5
    attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
    return attn @ v
""",
    },
    {
        "name": "conv2d_bias",
        "op_category": "convolution",
        "input_shapes": [[8, 64, 32, 32], [128, 64, 3, 3], [128]],
        "input_dtypes": ["float32", "float32", "float32"],
        "code": """
def op(x, weight, bias):
    return F.conv2d(x, weight, bias, padding=1)
""",
    },
    {
        "name": "group_norm",
        "op_category": "normalization",
        "input_shapes": [[8, 32, 64, 64], [32], [32]],
        "input_dtypes": ["float32", "float32", "float32"],
        "code": """
def op(x, weight, bias):
    return F.group_norm(x, 4, weight, bias)
""",
    },
    {
        "name": "log_sigmoid",
        "op_category": "activation",
        "input_shapes": [[2048, 512]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    return torch.log(torch.sigmoid(x))
""",
    },
    {
        "name": "bce_with_logits",
        "op_category": "loss",
        "input_shapes": [[64, 128], [64, 128]],
        "input_dtypes": ["float32", "float32"],
        "code": """
def op(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)
""",
    },
    {
        "name": "embedding_add_positions",
        "op_category": "embedding",
        "input_shapes": [[50000, 512], [2048, 512], [8, 128]],
        "input_dtypes": ["float32", "float32", "int64"],
        "code": """
def op(token_weight, pos_weight, indices):
    seq_len = indices.shape[1]
    pos_ids = torch.arange(seq_len, device=indices.device)
    return F.embedding(indices, token_weight) + F.embedding(pos_ids, pos_weight)
""",
    },
    {
        "name": "embedding_mean_pool",
        "op_category": "embedding",
        "input_shapes": [[32000, 512], [16, 64]],
        "input_dtypes": ["float32", "int64"],
        "code": """
def op(weight, indices):
    return F.embedding(indices, weight).mean(dim=1)
""",
    },
    {
        "name": "causal_attention",
        "op_category": "fused_attention",
        "input_shapes": [[2, 8, 128, 64], [2, 8, 128, 64], [2, 8, 128, 64]],
        "input_dtypes": ["float16", "float16", "float16"],
        "code": """
def op(q, k, v):
    scale = q.shape[-1] ** -0.5
    scores = q @ k.transpose(-2, -1) * scale
    L = q.shape[-2]
    mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float('-inf'))
    return torch.softmax(scores, dim=-1) @ v
""",
    },
    {
        "name": "depthwise_conv2d",
        "op_category": "convolution",
        "input_shapes": [[8, 32, 64, 64], [32, 1, 3, 3], [32]],
        "input_dtypes": ["float32", "float32", "float32"],
        "code": """
def op(x, weight, bias):
    return F.conv2d(x, weight, bias, padding=1, groups=x.shape[1])
""",
    },
    {
        "name": "conv1d_bias",
        "op_category": "convolution",
        "input_shapes": [[32, 256, 128], [512, 256, 3], [512]],
        "input_dtypes": ["float32", "float32", "float32"],
        "code": """
def op(x, weight, bias):
    return F.conv1d(x, weight, bias, padding=1)
""",
    },
    {
        "name": "fake_quantize_per_row",
        "op_category": "quantization",
        "input_shapes": [[256, 256], [256, 1], [256, 1]],
        "input_dtypes": ["float32", "float32", "float32"],
        "code": """
def op(x, scale, zero_point):
    x_int = torch.clamp(torch.round(x / scale) + zero_point, -128, 127)
    return (x_int - zero_point) * scale
""",
    },
    {
        "name": "mm_fp16",
        "op_category": "matmul",
        "input_shapes": [[128, 512], [512, 256]],
        "input_dtypes": ["float16", "float16"],
        "code": """
def op(a, b):
    return torch.mm(a, b)
""",
    },
    {
        "name": "linear_bias_fp16",
        "op_category": "matmul",
        "input_shapes": [[16, 256], [1024, 256], [1024]],
        "input_dtypes": ["float16", "float16", "float16"],
        "code": """
def op(x, w, b):
    return F.linear(x, w, b)
""",
    },
    {
        "name": "attention_scores",
        "op_category": "matmul",
        "input_shapes": [[4, 16, 64, 64], [4, 16, 64, 64]],
        "input_dtypes": ["float16", "float16"],
        "code": """
def op(q, k):
    return (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
""",
    },
    {
        "name": "l2_norm_rows",
        "op_category": "reduction",
        "input_shapes": [[128, 512]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    return x.norm(dim=-1)
""",
    },
    {
        "name": "max_reduction",
        "op_category": "reduction",
        "input_shapes": [[64, 256]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    return x.max(dim=-1).values
""",
    },
    {
        "name": "dot_product",
        "op_category": "reduction",
        "input_shapes": [[1048576], [1048576]],
        "input_dtypes": ["float32", "float32"],
        "code": """
def op(a, b):
    return (a * b).sum()
""",
    },
    {
        "name": "conv2d_strided",
        "op_category": "convolution",
        "input_shapes": [[8, 32, 64, 64], [64, 32, 3, 3], [64]],
        "input_dtypes": ["float32", "float32", "float32"],
        "code": """
def op(x, weight, bias):
    return F.conv2d(x, weight, bias, stride=2, padding=1)
""",
    },
    {
        "name": "conv_transpose2d",
        "op_category": "convolution",
        "input_shapes": [[8, 64, 32, 32], [64, 32, 2, 2], [32]],
        "input_dtypes": ["float32", "float32", "float32"],
        "code": """
def op(x, weight, bias):
    return F.conv_transpose2d(x, weight, bias, stride=2)
""",
    },
    {
        "name": "pointwise_conv2d",
        "op_category": "convolution",
        "input_shapes": [[8, 256, 32, 32], [512, 256, 1, 1], [512]],
        "input_dtypes": ["float32", "float32", "float32"],
        "code": """
def op(x, weight, bias):
    return F.conv2d(x, weight, bias)
""",
    },
    {
        "name": "dequantize_per_channel",
        "op_category": "quantization",
        "input_shapes": [[256, 512], [256], [256]],
        "input_dtypes": ["int32", "float32", "int32"],
        "code": """
def op(x, scale, zero_point):
    return (x - zero_point.unsqueeze(1)).float() * scale.unsqueeze(1)
""",
    },
    {
        "name": "dynamic_fake_quant",
        "op_category": "quantization",
        "input_shapes": [[512, 512]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    scale = x.abs().amax() / 127.0
    x_q = torch.clamp(torch.round(x / scale), -128, 127)
    return x_q * scale
""",
    },
    {
        "name": "quantize_per_row_symmetric",
        "op_category": "quantization",
        "input_shapes": [[256, 512]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    scale = x.abs().amax(dim=-1, keepdim=True) / 127.0
    x_q = torch.clamp(torch.round(x / scale), -128, 127)
    return x_q * scale
""",
    },
    {
        "name": "outer_product",
        "op_category": "matmul",
        "input_shapes": [[1024], [2048]],
        "input_dtypes": ["float32", "float32"],
        "code": """
def op(a, b):
    return torch.outer(a, b)
""",
    },
    {
        "name": "batched_mm_fp16",
        "op_category": "matmul",
        "input_shapes": [[8, 32, 128], [8, 128, 64]],
        "input_dtypes": ["float16", "float16"],
        "code": """
def op(a, b):
    return torch.bmm(a, b)
""",
    },
    {
        "name": "linear_relu",
        "op_category": "matmul",
        "input_shapes": [[64, 512], [1024, 512], [1024]],
        "input_dtypes": ["float32", "float32", "float32"],
        "code": """
def op(x, w, b):
    w = w * (w.shape[1] ** -0.5)
    return F.relu(F.linear(x, w, b))
""",
    },
    {
        "name": "rope_rotate",
        "op_category": "elementwise_chain",
        "input_shapes": [[4, 8, 128, 64], [128, 32], [128, 32]],
        "input_dtypes": ["float16", "float16", "float16"],
        "code": """
def op(x, cos, sin):
    x = x.float()
    cos = cos.float()
    sin = sin.float()
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
""",
    },
    {
        "name": "glu_gelu",
        "op_category": "elementwise_chain",
        "input_shapes": [[8, 512, 2048]],
        "input_dtypes": ["float16"],
        "code": """
def op(x):
    gate, val = x.chunk(2, dim=-1)
    return F.gelu(gate) * val
""",
    },
    {
        "name": "sdpa_builtin",
        "op_category": "fused_attention",
        "input_shapes": [[4, 8, 128, 64], [4, 8, 128, 64], [4, 8, 128, 64]],
        "input_dtypes": ["float16", "float16", "float16"],
        "code": """
def op(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)
""",
    },
    {
        "name": "batch_norm_manual",
        "op_category": "normalization",
        "input_shapes": [[16, 64, 32, 32], [64], [64]],
        "input_dtypes": ["float32", "float32", "float32"],
        "code": """
def op(x, weight, bias):
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
    x_hat = (x - mean) / torch.sqrt(var + 1e-5)
    return weight.view(1, -1, 1, 1) * x_hat + bias.view(1, -1, 1, 1)
""",
    },
    {
        "name": "rms_norm_fp16",
        "op_category": "normalization",
        "input_shapes": [[4, 512, 1024], [1024]],
        "input_dtypes": ["float16", "float16"],
        "code": """
def op(x, weight):
    x_float = x.float()
    rrms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + 1e-6)
    return x_float * rrms * weight.float()
""",
    },
    {
        "name": "gelu_sigmoid_approx",
        "op_category": "activation",
        "input_shapes": [[2048, 512]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    return x * torch.sigmoid(1.702 * x)
""",
    },
    {
        "name": "hardsigmoid_chain",
        "op_category": "elementwise_chain",
        "input_shapes": [[2048, 512]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    return F.hardsigmoid(x) * x
""",
    },
    {
        "name": "cross_entropy_manual",
        "op_category": "loss",
        "input_shapes": [[256, 1024], [256]],
        "input_dtypes": ["float32", "int64"],
        "code": """
def op(logits, targets):
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).mean()
""",
    },
    {
        "name": "kl_divergence",
        "op_category": "loss",
        "input_shapes": [[64, 512], [64, 512]],
        "input_dtypes": ["float32", "float32"],
        "code": """
def op(p, q):
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    return (p * (p.log() - q.log())).sum(dim=-1).mean()
""",
    },
    {
        "name": "embedding_scaled",
        "op_category": "embedding",
        "input_shapes": [[32000, 512], [8, 128]],
        "input_dtypes": ["float32", "int64"],
        "code": """
def op(weight, indices):
    return F.embedding(indices, weight) * (weight.shape[1] ** 0.5)
""",
    },
    {
        "name": "addmm",
        "op_category": "matmul",
        "input_shapes": [[256], [64, 512], [512, 256]],
        "input_dtypes": ["float32", "float32", "float32"],
        "code": """
def op(bias, a, b):
    return torch.addmm(bias, a, b)
""",
    },
    {
        "name": "scaled_bmm",
        "op_category": "matmul",
        "input_shapes": [[8, 64, 128], [8, 128, 64]],
        "input_dtypes": ["float16", "float16"],
        "code": """
def op(a, b):
    return torch.bmm(a, b) * (a.shape[-1] ** -0.5)
""",
    },
    {
        "name": "linear_silu_fp16",
        "op_category": "matmul",
        "input_shapes": [[32, 512], [2048, 512], [2048]],
        "input_dtypes": ["float16", "float16", "float16"],
        "code": """
def op(x, w, b):
    w = w.float() * (w.shape[1] ** -0.5)
    return F.silu(F.linear(x.float(), w, b.float()))
""",
    },
    {
        "name": "bias_gelu_fp16",
        "op_category": "elementwise_chain",
        "input_shapes": [[16, 256, 1024], [1024]],
        "input_dtypes": ["float16", "float16"],
        "code": """
def op(x, bias):
    return F.gelu(x + bias)
""",
    },
    {
        "name": "lerp_blend",
        "op_category": "elementwise_chain",
        "input_shapes": [[4, 512, 512], [4, 512, 512], [1]],
        "input_dtypes": ["float32", "float32", "float32"],
        "code": """
def op(a, b, t):
    return torch.lerp(a, b, t)
""",
    },
    {
        "name": "std_per_row",
        "op_category": "reduction",
        "input_shapes": [[64, 512]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    return x.std(dim=-1)
""",
    },
    {
        "name": "min_reduction",
        "op_category": "reduction",
        "input_shapes": [[64, 256]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    return x.min(dim=-1).values
""",
    },
    {
        "name": "qkv_packed_local_attention",
        "op_category": "fused_attention",
        "input_shapes": [[2, 128, 3, 8, 64]],
        "input_dtypes": ["float16"],
        "code": """
def op(qkv):
    q, k, v = qkv.unbind(dim=2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scale = q.shape[-1] ** -0.5
    scores = q @ k.transpose(-2, -1) * scale
    L = q.shape[-2]
    idx = torch.arange(L, device=q.device)
    local_causal_mask = (idx[None, :] > idx[:, None]) | (idx[None, :] < idx[:, None] - 32)
    scores = scores.masked_fill(local_causal_mask, float('-inf'))
    out = torch.softmax(scores, dim=-1) @ v
    return out.transpose(1, 2)
""",
    },
    {
        "name": "blockwise_activation_fake_quant",
        "op_category": "quantization",
        "input_shapes": [[128, 256]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    block_size = 64
    x_blocks = x.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    scale = x_blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
    q = torch.clamp(torch.round(x_blocks / scale), -128, 127)
    return (q * scale).reshape_as(x)
""",
    },
    {
        "name": "rms_norm_add",
        "op_category": "normalization",
        "input_shapes": [[4, 256, 1024], [4, 256, 1024], [1024]],
        "input_dtypes": ["float16", "float16", "float16"],
        "code": """
def op(x, residual, weight):
    h = x.float() + residual.float()
    inv_rms = torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return h * inv_rms * weight.float()
""",
    },
    {
        "name": "vocab_parallel_masked_embedding",
        "op_category": "embedding",
        "input_shapes": [[300, 256], [16, 64]],
        "input_dtypes": ["float32", "int64"],
        "code": """
def op(weight, input_ids):
    vocab_start = 50
    vocab_end = vocab_start + weight.shape[0]
    mask = (input_ids < vocab_start) | (input_ids >= vocab_end)
    local_ids = input_ids - vocab_start
    local_ids = local_ids.masked_fill(mask, 0)
    embeddings = F.embedding(local_ids, weight)
    return embeddings.masked_fill(mask.unsqueeze(-1), 0.0)
""",
    },
    {
        "name": "softplus_dt_bias",
        "op_category": "elementwise_chain",
        "input_shapes": [[8, 128, 16], [16]],
        "input_dtypes": ["float32", "float32"],
        "code": """
def op(dt, bias):
    return F.softplus(dt + bias)
""",
    },
    {
        "name": "kv_packed_gqa_attention",
        "op_category": "fused_attention",
        "input_shapes": [[2, 128, 8, 64], [2, 128, 2, 2, 64]],
        "input_dtypes": ["float16", "float16"],
        "code": """
def op(q, kv):
    k, v = kv.unbind(dim=2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    repeat = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    scores = q @ k.transpose(-2, -1) * (q.shape[-1] ** -0.5)
    return (torch.softmax(scores, dim=-1) @ v).transpose(1, 2)
""",
    },
    {
        "name": "alibi_attention_bias",
        "op_category": "fused_attention",
        "input_shapes": [[2, 8, 96, 64], [2, 8, 96, 64], [2, 8, 96, 64], [8]],
        "input_dtypes": ["float16", "float16", "float16", "float32"],
        "code": """
def op(q, k, v, slopes):
    Lq, Lk = q.shape[-2], k.shape[-2]
    q_pos = torch.arange(Lq, device=q.device)[:, None]
    k_pos = torch.arange(Lk, device=q.device)[None, :]
    distance = (q_pos - k_pos).abs().float()
    slopes = slopes.float().abs() / slopes.numel()
    bias = -slopes.view(1, -1, 1, 1) * distance.view(1, 1, Lq, Lk)
    scores = q.float() @ k.float().transpose(-2, -1) * (q.shape[-1] ** -0.5)
    scores = scores + bias
    return (torch.softmax(scores, dim=-1) @ v.float()).to(v.dtype)
""",
    },
    {
        "name": "cross_entropy_z_loss",
        "op_category": "loss",
        "input_shapes": [[256, 128], [256]],
        "input_dtypes": ["float32", "int64"],
        "code": """
def op(logits, labels):
    labels = labels.remainder(logits.shape[-1])
    lse = torch.logsumexp(logits, dim=-1)
    log_probs = logits - lse.unsqueeze(-1)
    nll = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    smooth = -log_probs.mean(dim=-1)
    return ((0.9 * nll) + (0.1 * smooth) + (1e-4 * lse.square())).mean()
""",
    },
    {
        "name": "bert_pretraining_loss",
        "op_category": "loss",
        "input_shapes": [[128, 256], [128], [32, 2], [32]],
        "input_dtypes": ["float32", "int64", "float32", "int64"],
        "code": """
def op(mlm_logits, mlm_labels, nsp_logits, nsp_labels):
    mlm_labels = mlm_labels.remainder(mlm_logits.shape[-1])
    nsp_labels = nsp_labels.remainder(nsp_logits.shape[-1])
    masked_lm_loss = F.cross_entropy(mlm_logits, mlm_labels)
    next_sentence_loss = F.cross_entropy(nsp_logits, nsp_labels)
    return masked_lm_loss.float() + next_sentence_loss.float()
""",
    },
    {
        "name": "fused_qk_rms_norm",
        "op_category": "normalization",
        "input_shapes": [[2, 8, 128, 64], [64], [2, 8, 128, 64], [64]],
        "input_dtypes": ["float16", "float16", "float16", "float16"],
        "code": """
def _rms_norm(x, weight, eps):
    x_float = x.float()
    inv_rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x_float * inv_rms * weight.float()

def op(q, q_weight, k, k_weight):
    q_norm = _rms_norm(q, q_weight, 1e-6)
    k_norm = _rms_norm(k, k_weight, 1e-6)
    return torch.cat([q_norm, k_norm], dim=-1)
""",
    },
    {
        "name": "asymmetric_int8_fake_quant",
        "op_category": "quantization",
        "input_shapes": [[512, 256]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    qmin, qmax = -128.0, 127.0
    x_min = x.amin()
    x_max = x.amax()
    scale = ((x_max - x_min) / (qmax - qmin)).clamp_min(1e-12)
    zero_point = torch.clamp(torch.round(qmin - x_min / scale), qmin, qmax)
    q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
    return (q - zero_point) * scale
""",
    },
    {
        "name": "int4_groupwise_dequant",
        "op_category": "quantization",
        "input_shapes": [[64, 8], [64, 1], [64, 1]],
        "input_dtypes": ["int32", "float32", "int32"],
        "code": """
def _unpack_int4(packed):
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    values = (packed.unsqueeze(-1) >> shifts) & 0xF
    return values.reshape(packed.shape[0], packed.shape[1] * 8).float()

def op(packed_weight, scales, zeros):
    q = _unpack_int4(packed_weight)
    scale = scales.abs().clamp_min(1e-6)
    zero = zeros.remainder(16).float()
    return (q - zero) * scale
""",
    },
    {
        "name": "token_type_task_position_embedding",
        "op_category": "embedding",
        "input_shapes": [[100, 256], [2, 256], [1, 256], [128, 256], [8, 128], [8, 128]],
        "input_dtypes": ["float32", "float32", "float32", "float32", "int64", "int64"],
        "code": """
def op(word_weight, token_type_weight, task_weight, pos_weight, input_ids, token_type_ids):
    input_ids = input_ids.remainder(word_weight.shape[0])
    token_type_ids = token_type_ids.remainder(token_type_weight.shape[0])
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    position_ids = position_ids.expand_as(input_ids)
    task_ids = torch.zeros_like(input_ids)
    embeddings = F.embedding(input_ids, word_weight)
    embeddings = embeddings + F.embedding(token_type_ids, token_type_weight)
    embeddings = embeddings + F.embedding(task_ids, task_weight)
    embeddings = embeddings + F.embedding(position_ids, pos_weight)
    return F.layer_norm(embeddings, (embeddings.shape[-1],))
""",
    },
    {
        "name": "prefix_multimodal_embedding_merge",
        "op_category": "embedding",
        "input_shapes": [[100, 256], [2, 64], [2, 8, 256]],
        "input_dtypes": ["float32", "int64", "float32"],
        "code": """
def op(text_weight, input_ids, image_embeds):
    input_ids = input_ids.remainder(text_weight.shape[0])
    text_embeds = F.embedding(input_ids, text_weight)
    n_image_tokens = image_embeds.shape[1]
    return torch.cat([image_embeds.to(text_embeds.dtype), text_embeds[:, n_image_tokens:, :]], dim=1)
""",
    },
    {
        "name": "swiglu_oai",
        "op_category": "activation",
        "input_shapes": [[2048, 1024]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(max=7.0)
    up = up.clamp(min=-7.0, max=7.0)
    glu = gate * torch.sigmoid(1.702 * gate)
    return (up + 1.0) * glu
""",
    },
    {
        "name": "fatrelu_and_mul",
        "op_category": "activation",
        "input_shapes": [[4, 256, 1024]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    d = x.shape[-1] // 2
    gate = x[..., :d]
    up = x[..., d:]
    gate = F.threshold(gate, 0.0, 0.0)
    return gate * up
""",
    },
    {
        "name": "causal_depthwise_conv1d_silu",
        "op_category": "convolution",
        "input_shapes": [[8, 128, 256], [128, 4], [128]],
        "input_dtypes": ["float32", "float32", "float32"],
        "code": """
def op(x, weight, bias):
    dim, width = weight.shape
    y = F.conv1d(x.to(weight.dtype), weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    y = y[..., :x.shape[-1]]
    return F.silu(y).to(x.dtype)
""",
    },
    {
        "name": "softcap_attention",
        "op_category": "fused_attention",
        "input_shapes": [[2, 8, 96, 64], [2, 8, 96, 64], [2, 8, 96, 64]],
        "input_dtypes": ["float16", "float16", "float16"],
        "code": """
def op(q, k, v):
    softcap = 30.0
    scores = q @ k.transpose(-2, -1) * (q.shape[-1] ** -0.5)
    scores = torch.tanh(scores.float() / softcap) * softcap
    attn = torch.softmax(scores, dim=-1).to(v.dtype)
    return attn @ v
""",
    },
    {
        "name": "ignore_index_cross_entropy",
        "op_category": "loss",
        "input_shapes": [[256, 128], [256]],
        "input_dtypes": ["float32", "int64"],
        "code": """
def op(logits, labels):
    labels = labels.remainder(logits.shape[-1])
    ignore = labels == 0
    safe_labels = labels.masked_fill(ignore, 0)
    logits = logits.float()
    lse = torch.logsumexp(logits, dim=-1)
    target_logits = logits.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
    losses = (lse - target_logits).masked_fill(ignore, 0.0)
    denom = (~ignore).sum().clamp_min(1)
    return losses.sum() / denom
""",
    },
    {
        "name": "parallel_residual_layer_norm",
        "op_category": "normalization",
        "input_shapes": [[4, 128, 512], [4, 128, 512], [4, 128, 512], [512], [512], [512], [512]],
        "input_dtypes": ["float32", "float32", "float32", "float32", "float32", "float32", "float32"],
        "code": """
def _layer_norm(x, weight, bias, eps):
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    var = centered.pow(2).mean(dim=-1, keepdim=True)
    return centered * torch.rsqrt(var + eps) * weight + bias

def op(x0, x1, residual, weight0, bias0, weight1, bias1):
    h = x0 + x1 + residual
    z0 = _layer_norm(h, weight0, bias0, 1e-5)
    z1 = _layer_norm(h, weight1, bias1, 1e-5)
    return torch.cat([z0, z1], dim=-1)
""",
    },
    {
        "name": "swiglustep_and_mul",
        "op_category": "activation",
        "input_shapes": [[4, 256, 1024]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    gate, up = x.chunk(2, dim=-1)
    gate = F.silu(gate).clamp(max=7.0)
    up = up.clamp(min=-7.0, max=7.0)
    return gate * up
""",
    },
    {
        "name": "projected_position_embedding",
        "op_category": "embedding",
        "input_shapes": [[100, 128], [256, 128], [64, 256], [8, 64]],
        "input_dtypes": ["float32", "float32", "float32", "int64"],
        "code": """
def op(word_weight, proj_weight, pos_weight, input_ids):
    input_ids = input_ids.remainder(word_weight.shape[0])
    word = F.embedding(input_ids, word_weight)
    word = F.linear(word, proj_weight)
    pos_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
    pos_ids = pos_ids.remainder(pos_weight.shape[0])
    return word + F.embedding(pos_ids, pos_weight)
""",
    },
    {
        "name": "attention_with_sink",
        "op_category": "fused_attention",
        "input_shapes": [[2, 8, 96, 64], [2, 8, 96, 64], [2, 8, 96, 64], [8]],
        "input_dtypes": ["float16", "float16", "float16", "float32"],
        "code": """
def op(q, k, v, sink_logits):
    scores = (q @ k.transpose(-2, -1)).float() * (q.shape[-1] ** -0.5)
    sink = sink_logits.float().view(1, -1, 1, 1)
    row_max = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
    unnormalized = torch.exp(scores - row_max)
    normalizer = unnormalized.sum(dim=-1, keepdim=True) + torch.exp(sink - row_max)
    attn = (unnormalized / normalizer).to(v.dtype)
    return attn @ v
""",
    },
    {
        "name": "shifted_lm_cross_entropy",
        "op_category": "loss",
        "input_shapes": [[4, 64, 128], [4, 64]],
        "input_dtypes": ["float32", "int64"],
        "code": """
def op(logits, input_ids):
    shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = input_ids[:, 1:].reshape(-1).remainder(logits.shape[-1])
    return F.cross_entropy(shift_logits.float(), shift_labels)
""",
    },
    {
        "name": "codebook_vector_quantize",
        "op_category": "quantization",
        "input_shapes": [[1024, 4], [16, 4]],
        "input_dtypes": ["float32", "float32"],
        "code": """
def op(x, codebook):
    scale = x.abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)
    normalized = x / scale
    codebook = codebook.to(x.dtype)
    distances = (normalized.unsqueeze(1) - codebook.unsqueeze(0)).pow(2).sum(dim=-1)
    indices = distances.argmin(dim=-1)
    quantized = codebook.index_select(0, indices.reshape(-1)).reshape_as(x)
    return quantized * scale
""",
    },
    {
        "name": "conformer_depthwise_conv_module",
        "op_category": "convolution",
        "input_shapes": [[4, 128, 256], [1024, 256, 1], [512, 1, 33], [256, 512, 1]],
        "input_dtypes": ["float32", "float32", "float32", "float32"],
        "code": """
def op(x, pointwise_in, depthwise, pointwise_out):
    residual = x
    h = F.layer_norm(x, (x.shape[-1],))
    h = h.transpose(1, 2)
    pointwise_in = pointwise_in * (pointwise_in.shape[1] ** -0.5)
    depthwise = depthwise * (depthwise.shape[-1] ** -0.5)
    pointwise_out = pointwise_out * (pointwise_out.shape[1] ** -0.5)
    h = F.conv1d(h, pointwise_in)
    h = F.glu(h, dim=1)
    padding = (depthwise.shape[-1] - 1) // 2
    h = F.conv1d(h, depthwise, padding=padding, groups=h.shape[1])
    h = h.transpose(1, 2)
    h = F.layer_norm(h, (h.shape[-1],))
    h = h * torch.sigmoid(h)
    h = F.conv1d(h.transpose(1, 2), pointwise_out)
    return h.transpose(1, 2) + residual
""",
    },
    {
        "name": "grouped_gated_rms_norm",
        "op_category": "normalization",
        "input_shapes": [[4, 128, 512], [4, 128, 512], [512], [512]],
        "input_dtypes": ["float16", "float16", "float32", "float32"],
        "code": """
def op(x, gate, weight, bias):
    group_size = 64
    x_float = x.float()
    grouped = x_float.reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size)
    inv_rms = torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + 1e-5)
    normalized = (grouped * inv_rms).reshape_as(x_float)
    out = normalized * weight.float() + bias.float()
    return (out * F.silu(gate.float())).to(x.dtype)
""",
    },
    {
        "name": "cu_seqlens_from_mask",
        "op_category": "reduction",
        "input_shapes": [[32, 128], [32, 128]],
        "input_dtypes": ["bool", "bool"],
        "code": """
def op(attention_mask, padding_mask):
    all_masks = attention_mask | padding_mask
    seqlens = all_masks.sum(dim=-1, dtype=torch.int32)
    prefix = torch.cumsum(seqlens, dim=0, dtype=torch.int32)
    zero = torch.zeros(1, device=attention_mask.device, dtype=torch.int32)
    return torch.cat([zero, prefix], dim=0)
""",
    },
    {
        "name": "topk_sparse_attention",
        "op_category": "fused_attention",
        "input_shapes": [[2, 64, 8, 64], [2, 64, 8, 64], [2, 64, 8, 64], [2, 64, 16]],
        "input_dtypes": ["float16", "float16", "float16", "int64"],
        "code": """
def op(q, k, v, gather_indices):
    gather_indices = gather_indices.remainder(k.shape[1])
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("bthd,bshd->bhts", q.float() * scale, k.float())
    key_ids = torch.arange(k.shape[1], device=q.device)
    topk_mask = (gather_indices.unsqueeze(-1) == key_ids.view(1, 1, 1, -1)).any(dim=2)
    scores = scores.masked_fill(~topk_mask.unsqueeze(1), float("-inf"))
    attn = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.einsum("bhts,bshd->bthd", attn, v)
""",
    },
    {
        "name": "fp8_pow2_fake_quant",
        "op_category": "quantization",
        "input_shapes": [[512, 256]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    fp8_max = 448.0
    amax = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = fp8_max / amax
    scale = torch.exp2(torch.floor(torch.log2(scale)))
    q = torch.clamp(torch.round(x.float() * scale), -fp8_max, fp8_max)
    return q / scale
""",
    },
    {
        "name": "gelu_sparse_topk_mul",
        "op_category": "activation",
        "input_shapes": [[4, 256, 1024]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    d = x.shape[-1] // 2
    gate, up = x[..., :d], x[..., d:]
    mean = gate.mean(dim=-1, keepdim=True)
    std = gate.std(dim=-1, keepdim=True, unbiased=False)
    sparse_gate = F.relu(gate - (mean + 0.25 * std))
    return F.gelu(sparse_gate) * up
""",
    },
    {
        "name": "relu_squared",
        "op_category": "activation",
        "input_shapes": [[2048, 512]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    y = F.relu(x)
    return y * y
""",
    },
    {
        "name": "clip_patch_class_embedding",
        "op_category": "convolution",
        "input_shapes": [[4, 3, 32, 32], [256, 3, 4, 4], [256], [65, 256]],
        "input_dtypes": ["float32", "float32", "float32", "float32"],
        "code": """
def op(pixel_values, patch_weight, class_embedding, position_weight):
    patches = F.conv2d(pixel_values, patch_weight, stride=4)
    patches = patches.flatten(2).transpose(1, 2)
    class_token = class_embedding.view(1, 1, -1).expand(pixel_values.shape[0], 1, -1)
    embeddings = torch.cat([class_token.to(patches.dtype), patches], dim=1)
    position_ids = torch.arange(embeddings.shape[1], device=pixel_values.device)
    return embeddings + F.embedding(position_ids, position_weight).unsqueeze(0)
""",
    },
    {
        "name": "causal_conv2d_time",
        "op_category": "convolution",
        "input_shapes": [[4, 16, 24, 64], [32, 16, 3, 3], [32]],
        "input_dtypes": ["float32", "float32", "float32"],
        "code": """
def op(x, weight, bias):
    left_padding = weight.shape[-1] - 1
    padded = F.pad(x, (left_padding, 0, 0, 0))
    return F.conv2d(padded, weight, bias)
""",
    },
    {
        "name": "bert_decoded_type_position_norm",
        "op_category": "embedding",
        "input_shapes": [[128, 256], [2, 256], [128, 256], [8, 128], [256], [256]],
        "input_dtypes": ["float32", "float32", "float32", "int64", "float32", "float32"],
        "code": """
def op(word_weight, token_type_weight, position_weight, input_ids, norm_weight, norm_bias):
    token_type_shift = 6
    token_type_ids = (input_ids >> token_type_shift).remainder(token_type_weight.shape[0])
    token_ids = input_ids.bitwise_and((1 << token_type_shift) - 1).remainder(word_weight.shape[0])
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    embeddings = F.embedding(token_ids, word_weight)
    embeddings = embeddings + F.embedding(token_type_ids, token_type_weight)
    embeddings = embeddings + F.embedding(position_ids.expand_as(input_ids), position_weight)
    return F.layer_norm(embeddings, (embeddings.shape[-1],), norm_weight, norm_bias)
""",
    },
    {
        "name": "padded_tied_lm_embedding",
        "op_category": "embedding",
        "input_shapes": [[120, 256], [8, 32]],
        "input_dtypes": ["float32", "int64"],
        "code": """
def op(word_weight, input_ids):
    padded_weight = F.pad(word_weight, (0, 0, 0, 8))
    input_ids = input_ids.remainder(padded_weight.shape[0])
    hidden = F.embedding(input_ids, padded_weight)
    return F.linear(hidden, padded_weight)
""",
    },
    {
        "name": "maxsim_margin_loss",
        "op_category": "loss",
        "input_shapes": [[8, 16, 128], [8, 48, 128], [8, 48, 128]],
        "input_dtypes": ["float32", "float32", "float32"],
        "code": """
def _maxsim_score(query, doc):
    scale = query.shape[-1] ** -0.5
    token_scores = torch.bmm(query.float() * scale, doc.float().transpose(1, 2))
    return token_scores.amax(dim=-1).mean(dim=-1)

def op(query, positive_doc, negative_doc):
    positive = _maxsim_score(query, positive_doc)
    negative = _maxsim_score(query, negative_doc)
    return F.relu(0.2 - positive + negative).mean()
""",
    },
    {
        "name": "negative_sqnr_loss",
        "op_category": "loss",
        "input_shapes": [[1024, 256], [1024, 256]],
        "input_dtypes": ["float32", "float32"],
        "code": """
def op(actual, ref):
    actual = actual.float()
    ref = ref.float()
    signal = ref.pow(2).mean().clamp_min(1e-12)
    noise = (ref - actual).pow(2).mean().clamp_min(1e-12)
    sqnr = 10.0 * torch.log10(signal / noise)
    return -sqnr
""",
    },
    {
        "name": "layer_norm_linear",
        "op_category": "normalization",
        "input_shapes": [[16, 128, 512], [512], [512], [256, 512], [256]],
        "input_dtypes": ["float32", "float32", "float32", "float32", "float32"],
        "code": """
def op(x, norm_weight, norm_bias, linear_weight, linear_bias):
    x_float = x.float()
    mean = x_float.mean(dim=-1, keepdim=True)
    var = (x_float - mean).pow(2).mean(dim=-1, keepdim=True)
    normalized = (x_float - mean) * torch.rsqrt(var + 1e-6)
    normalized = normalized * norm_weight.float() + norm_bias.float()
    linear_weight = linear_weight * (linear_weight.shape[1] ** -0.5)
    return F.linear(normalized.to(linear_weight.dtype), linear_weight, linear_bias)
""",
    },
    {
        "name": "weight_only_layer_norm",
        "op_category": "normalization",
        "input_shapes": [[4, 128, 512], [512]],
        "input_dtypes": ["float32", "float32"],
        "code": """
def op(hidden_states, weight):
    input_dtype = hidden_states.dtype
    h = hidden_states.float()
    mean = h.mean(dim=-1, keepdim=True)
    variance = (h - mean).pow(2).mean(dim=-1, keepdim=True)
    h = (h - mean) * torch.rsqrt(variance + 1e-5)
    return (h * weight.float()).to(input_dtype)
""",
    },
    {
        "name": "qv_augmented_attention",
        "op_category": "fused_attention",
        "input_shapes": [[2, 64, 8, 64], [2, 64, 8, 64], [2, 64, 8, 32], [2, 64, 8, 32]],
        "input_dtypes": ["float16", "float16", "float16", "float16"],
        "code": """
def op(q, k, v, qv):
    scale = (q.shape[-1] + qv.shape[-1]) ** -0.5
    qk_scores = torch.einsum("bthd,bshd->bhts", q.float() * scale, k.float())
    qv_scores = torch.einsum("bthd,bshd->bhts", qv.float() * scale, v.float())
    attn = torch.softmax(qk_scores + qv_scores, dim=-1).to(v.dtype)
    return torch.einsum("bhts,bshd->bthd", attn, v)
""",
    },
    {
        "name": "attention_logsumexp",
        "op_category": "reduction",
        "input_shapes": [[2, 8, 96, 64], [2, 8, 96, 64]],
        "input_dtypes": ["float16", "float16"],
        "code": """
def op(q, k):
    scores = (q.float() @ k.float().transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    return torch.logsumexp(scores, dim=-1)
""",
    },
    {
        "name": "xielu_activation",
        "op_category": "activation",
        "input_shapes": [[2048, 512]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    alpha_p = 0.8
    alpha_n = 0.8
    beta = 0.5
    eps = x.new_tensor(-1e-6)
    positive = alpha_p * x * x + beta * x
    negative = (torch.expm1(torch.minimum(x, eps)) - x) * alpha_n + beta * x
    return torch.where(x > 0, positive, negative)
""",
    },
    {
        "name": "nemo_dw_striding_subsample",
        "op_category": "convolution",
        "input_shapes": [[4, 64, 80], [32, 1, 3, 3], [32], [32, 1, 3, 3], [32], [32, 32, 1, 1], [32], [128, 640], [128]],
        "input_dtypes": ["float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32"],
        "code": """
def op(x, conv_weight, conv_bias, depthwise_weight, depthwise_bias, pointwise_weight, pointwise_bias, out_weight, out_bias):
    conv_weight = conv_weight * ((conv_weight.shape[1] * conv_weight.shape[2] * conv_weight.shape[3]) ** -0.5)
    depthwise_weight = depthwise_weight * ((depthwise_weight.shape[2] * depthwise_weight.shape[3]) ** -0.5)
    pointwise_weight = pointwise_weight * (pointwise_weight.shape[1] ** -0.5)
    out_weight = out_weight * (out_weight.shape[1] ** -0.5)
    h = x.unsqueeze(1)
    h = F.relu(F.conv2d(h, conv_weight, conv_bias, stride=2, padding=1))
    h = F.relu(F.conv2d(h, depthwise_weight, depthwise_bias, stride=2, padding=1, groups=h.shape[1]))
    h = F.conv2d(h, pointwise_weight, pointwise_bias)
    batch, channels, time, freq = h.shape
    h = h.transpose(1, 2).reshape(batch, time, channels * freq)
    return F.linear(h, out_weight, out_bias)
""",
    },
    {
        "name": "column_parallel_embedding_slice",
        "op_category": "embedding",
        "input_shapes": [[1000, 512], [8, 64]],
        "input_dtypes": ["float32", "int64"],
        "code": """
def op(weight, input_ids):
    input_ids = input_ids.remainder(weight.shape[0])
    local_dim = weight.shape[1] // 2
    local_weight = weight[:, :local_dim].contiguous()
    return F.embedding(input_ids, local_weight)
""",
    },
    {
        "name": "parallel_vocab_cross_entropy",
        "op_category": "loss",
        "input_shapes": [[256, 128], [256]],
        "input_dtypes": ["float32", "int64"],
        "code": """
def op(local_logits, labels):
    vocab_start = 32
    labels = labels.remainder(local_logits.shape[-1] + 64)
    in_shard = (labels >= vocab_start) & (labels < vocab_start + local_logits.shape[-1])
    local_labels = (labels - vocab_start).clamp(0, local_logits.shape[-1] - 1)
    logits = local_logits.float() * 0.5
    lse = torch.logsumexp(logits, dim=-1)
    target_logits = logits.gather(1, local_labels.unsqueeze(1)).squeeze(1)
    nll = (lse - target_logits).masked_fill(~in_shard, 0.0)
    smooth = 0.05 * (lse - logits.mean(dim=-1))
    denom = in_shard.sum().clamp_min(1)
    return nll.sum() / denom + smooth.mean()
""",
    },
    {
        "name": "dropout_add_layer_norm_rowscale",
        "op_category": "normalization",
        "input_shapes": [[4, 128, 512], [4, 128, 512], [4, 128], [512], [512]],
        "input_dtypes": ["float32", "float32", "float32", "float32", "float32"],
        "code": """
def op(x, residual, rowscale, weight, bias):
    h = x.float() * rowscale.float().unsqueeze(-1) + residual.float()
    mean = h.mean(dim=-1, keepdim=True)
    centered = h - mean
    var = centered.pow(2).mean(dim=-1, keepdim=True)
    normalized = centered * torch.rsqrt(var + 1e-5)
    return normalized * weight.float() + bias.float()
""",
    },
    {
        "name": "chunked_local_attention",
        "op_category": "fused_attention",
        "input_shapes": [[2, 64, 8, 64], [2, 64, 8, 64], [2, 64, 8, 64]],
        "input_dtypes": ["float16", "float16", "float16"],
        "code": """
def op(q, k, v):
    chunk = 16
    scores = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) * (q.shape[-1] ** -0.5)
    positions = torch.arange(q.shape[1], device=q.device)
    same_chunk = (positions[:, None] // chunk) == (positions[None, :] // chunk)
    causal = positions[None, :] <= positions[:, None]
    scores = scores.masked_fill(~(same_chunk & causal).view(1, 1, q.shape[1], k.shape[1]), float("-inf"))
    attn = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.einsum("bhts,bshd->bthd", attn, v)
""",
    },
    {
        "name": "splade_log1p_max_pool",
        "op_category": "reduction",
        "input_shapes": [[4, 64, 512], [4, 64]],
        "input_dtypes": ["float32", "int64"],
        "code": """
def op(logits, token_ids):
    positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
    ids = token_ids + 4
    ids = torch.where(positions == 0, torch.full_like(ids, 101), ids)
    ids = torch.where(positions == token_ids.shape[1] - 1, torch.full_like(ids, 102), ids)
    is_cls = (positions == 0) & (ids == 101)
    is_sep = (positions == token_ids.shape[1] - 1) & (ids == 102)
    keep = ~(is_cls | is_sep)
    scores = torch.log1p(F.relu(logits.float()))
    scores = scores.masked_fill(~keep.unsqueeze(-1), 0.0)
    return scores.amax(dim=1)
""",
    },
    {
        "name": "interleaved_rotary_embedding",
        "op_category": "elementwise_chain",
        "input_shapes": [[2, 128, 8, 64], [128, 16], [128, 16]],
        "input_dtypes": ["float16", "float16", "float16"],
        "code": """
def op(x, cos, sin):
    x = x.float()
    cos = cos.float()
    sin = sin.float()
    rotary_dim = cos.shape[-1] * 2
    x_ro = x[..., :rotary_dim]
    x_even = x_ro[..., ::2]
    x_odd = x_ro[..., 1::2]
    cos = cos.view(1, cos.shape[0], 1, cos.shape[1])
    sin = sin.view(1, sin.shape[0], 1, sin.shape[1])
    rotated = torch.stack([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1).flatten(-2)
    return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)
""",
    },
    {
        "name": "sigmoid_glu_split",
        "op_category": "elementwise_chain",
        "input_shapes": [[8, 512, 2048]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    value, gate = x.chunk(2, dim=-1)
    return value * torch.sigmoid(gate)
""",
    },
    {
        "name": "fp8_blockwise_weight_fake_quant",
        "op_category": "quantization",
        "input_shapes": [[256, 256]],
        "input_dtypes": ["float32"],
        "code": """
def op(weight):
    block_size = 64
    rows = weight.shape[0] // block_size
    cols = weight.shape[1] // block_size
    blocks = weight.reshape(rows, block_size, cols, block_size).permute(0, 2, 1, 3)
    scale = blocks.float().abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1e-12) / 448.0
    q = torch.clamp(torch.round(blocks.float() / scale), -448.0, 448.0)
    return (q * scale).permute(0, 2, 1, 3).reshape_as(weight)
""",
    },
    {
        "name": "int8_asym_per_token_fake_quant",
        "op_category": "quantization",
        "input_shapes": [[32, 128, 512]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    qmin, qmax = -128.0, 127.0
    x = x.double()
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    scale = ((x_max - x_min) / (qmax - qmin)).clamp_min(1e-12)
    zero_point = torch.clamp(torch.round(qmin - x_min / scale), qmin, qmax)
    q = torch.clamp(torch.floor(x / scale + zero_point + 0.5), qmin, qmax)
    return ((q - zero_point) * scale).float()
""",
    },
]
