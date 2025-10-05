from fla.models.delta_net.configuration_delta_net import DeltaNetConfig


cfg = DeltaNetConfig(
    # core architecture
    attn_mode = "chunk",
    hidden_size = 2048,
    expand_k = 1.0,
    expand_v = 1.0,
    use_gate = False,
    use_short_conv = True,
    conv_size = 4,
    use_beta = True,
    use_output_norm = True,
    num_heads = 16,
    qk_norm = 'l2',
    qk_activation = 'silu',
    max_position_embeddings = 2048,
    hidden_ratio = 4,
    intermediate_size = None,
    hidden_act = "swish",
    num_hidden_layers = 24,
    norm_eps = 1e-6,
    attn = None,
    use_cache = True,
    pad_token_id = None,
    bos_token_id = 1,
    eos_token_id = 2,
    tie_word_embeddings = False,
    initializer_range = 0.02,
    fuse_norm = True,
    fuse_swiglu = True,
    fuse_cross_entropy = True,
    fuse_linear_cross_entropy = False,
    use_l2warp = False,
    vocab_size = 32000
)