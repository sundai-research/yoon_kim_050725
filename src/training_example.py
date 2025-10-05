from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from fla.models.transformer.configuration_transformer import TransformerConfig
from fla.models.transformer.modeling_transformer import TransformerForCausalLM

from model_cfgs.transformer_cfg import cfg as transformer_cfg
from data_gen import DataConfig, get_dataloaders
from hack_utils import non_shifting_loss, compute_accuracy

def train(
    model: TransformerForCausalLM,
    train_dl: DataLoader,
    optimizer: optim.Optimizer,
    max_epochs: int,
):
    model.train()
    for epoch in range(max_epochs):
        for inputs, targets in tqdm(train_dl, desc=f"Train Epoch {epoch}/{max_epochs}"):
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            optimizer.zero_grad()
            output = model(inputs, attention_mask=torch.ones_like(inputs))
            logits = output.logits
            loss = non_shifting_loss(logits, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    return model

def eval(model: TransformerForCausalLM, test_dl: DataLoader):
    model.eval()
    all_logits = []
    all_targets = []
    for inputs, targets in test_dl:
        inputs, targets = inputs.to(model.device), targets.to(model.device)
        output = model(inputs)
        logits = output.logits
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return compute_accuracy(all_logits, all_targets)

if __name__ == "__main__":

    data_cfg = DataConfig(
        num_train_examples=100_000,
        num_test_examples=3_000,
        input_seq_len=64,
        vocab_size=65,
        batch_size=256,
        num_kv_pairs=16,
        train_power_a=0.01,
        test_power_a=0.01,
        random_non_queries=False,
        seed=37,
    )
    transformer_cfg = TransformerConfig(
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


    train_dl, test_dl = get_dataloaders(data_cfg)

    model_cfg = transformer_cfg
    model = TransformerForCausalLM(model_cfg).to(torch.device("cuda:0")).to(torch.bfloat16)
    torch.compile(model)

    optimizer = optim.AdamW(model.parameters(), 
                            lr=1e-4, 
                            weight_decay=1e-6)

    # from ipdb import set_trace; set_trace()
    model = train(model, train_dl, optimizer, 30)
    accuracy = eval(model, test_dl)
    print(f"Accuracy: {accuracy}")
    # from ipdb import set_trace; set_trace()