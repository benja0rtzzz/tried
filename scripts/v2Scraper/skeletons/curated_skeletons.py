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
        "op_category": "reduction",
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
    # ------------------------------------------------------------------ #
    # convolution (cont.)                                                  #
    # ------------------------------------------------------------------ #
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
    # ------------------------------------------------------------------ #
    # quantization (cont.)                                                 #
    # ------------------------------------------------------------------ #
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
    # ------------------------------------------------------------------ #
    # matmul (cont.)                                                       #
    # ------------------------------------------------------------------ #
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
    return F.relu(F.linear(x, w, b))
""",
    },
    # ------------------------------------------------------------------ #
    # elementwise_chain (cont.)                                            #
    # ------------------------------------------------------------------ #
    {
        "name": "rope_rotate",
        "op_category": "elementwise_chain",
        "input_shapes": [[4, 8, 128, 64], [128, 32], [128, 32]],
        "input_dtypes": ["float16", "float16", "float16"],
        "code": """
def op(x, cos, sin):
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
    # ------------------------------------------------------------------ #
    # fused_attention (cont.)                                              #
    # ------------------------------------------------------------------ #
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
    # ------------------------------------------------------------------ #
    # normalization (cont.)                                                #
    # ------------------------------------------------------------------ #
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
    rrms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype)
    return x * rrms * weight
""",
    },
    # ------------------------------------------------------------------ #
    # activation (cont.)                                                   #
    # ------------------------------------------------------------------ #
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
        "op_category": "activation",
        "input_shapes": [[2048, 512]],
        "input_dtypes": ["float32"],
        "code": """
def op(x):
    return F.hardsigmoid(x) * x
""",
    },
    # ------------------------------------------------------------------ #
    # loss (cont.)                                                         #
    # ------------------------------------------------------------------ #
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
    # ------------------------------------------------------------------ #
    # embedding (cont.)                                                    #
    # ------------------------------------------------------------------ #
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
    # ------------------------------------------------------------------ #
    # matmul (cont.)                                                       #
    # ------------------------------------------------------------------ #
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
    return F.silu(F.linear(x, w, b))
""",
    },
    # ------------------------------------------------------------------ #
    # elementwise_chain (cont.)                                            #
    # ------------------------------------------------------------------ #
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
    # ------------------------------------------------------------------ #
    # reduction (cont.)                                                    #
    # ------------------------------------------------------------------ #
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
]
