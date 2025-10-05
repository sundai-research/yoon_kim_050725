from fla.models.transformer.configuration_transformer import TransformerConfig
from fla.models.gla.configuration_gla import GLAConfig

cfg = TransformerConfig(
    # core architecture
    vocab_size=769,
    hidden_size=256,
    num_hidden_layers=2,
    num_heads=1,
    num_kv_heads=None,
    max_position_embeddings=1024,
    rope_theta=10000.0,
    qkv_bias=False,
    qk_norm=False,
    window_size=None,

    # MLP (SwiGLU)
    hidden_ratio=4,
    intermediate_size=1024,
    hidden_act="swish",

    # norms and numerics
    norm_eps=1e-6,
    elementwise_affine=True,
    fuse_norm=False,
    fuse_swiglu=False,
    fuse_cross_entropy=False,
    fuse_linear_cross_entropy=False,
    use_l2warp=False,

    # runtime semantics
    use_cache=False,
    tie_word_embeddings=False,
)

aldo_cfg = TransformerConfig(
    # core architecture
    vocab_size=65,
    hidden_size=256,
    num_hidden_layers=2,
    num_heads=1,
    num_kv_heads=None,
    max_position_embeddings=1024,
    rope_theta=10000.0,
    qkv_bias=False,
    qk_norm=False,
    window_size=None,

    # MLP (SwiGLU)
    hidden_ratio=4,
    intermediate_size=1024,
    hidden_act="swish",

    # norms and numerics
    norm_eps=1e-6,
    elementwise_affine=True,
    fuse_norm=False,
    fuse_swiglu=False,
    fuse_cross_entropy=False,
    fuse_linear_cross_entropy=False,
    use_l2warp=False,

    # runtime semantics
    use_cache=False,
    tie_word_embeddings=False,
)

gla_cfg = GLAConfig(
    vocab_size=769,
    hidden_size=256,
    num_hidden_layers=2,
    num_heads=1,
    num_kv_heads=None,
    max_position_embeddings=1024,
)