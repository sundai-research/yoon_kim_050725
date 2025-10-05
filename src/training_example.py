from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from fla.models.transformer.modeling_transformer import TransformerForCausalLM
from fla.models.delta_net.modeling_delta_net import DeltaNetForCausalLM
from fla.models.delta_net.configuration_delta_net import DeltaNetConfig

from model_cfgs.transformer_cfg import cfg as transformer_cfg
from model_cfgs.deltanet_cfg import cfg as deltanet_cfg


from data_gen import DataConfig, get_dataloaders
from hack_utils import non_shifting_loss, compute_accuracy

def train(
    model: DeltaNetForCausalLM,
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
        accuracy = eval(model, test_dl)
        print(f"Accuracy: {accuracy}")
        if accuracy > 0.995:
            print("Early stopping, saturation reached.")
            break
        model.train()
    return model

def eval(model: DeltaNetForCausalLM, test_dl: DataLoader):
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

   
    kv_options = (64, 256, 512)  # three settings

    data_configs = []
    for kv in kv_options:
        cfg = DataConfig(
            num_train_examples=100_000,
            num_test_examples=3_000,
            input_seq_len=4 * kv,
            vocab_size=4 * kv + 1,
            batch_size=256,
            num_kv_pairs=kv,
            train_power_a=0.01,
            test_power_a=0.01,
            random_non_queries=False,
            seed=37,
            cache_path = f"data_{kv}.pt"
        )
        data_configs.append(cfg)

    for data_cfg in data_configs:
        vocab_size = data_cfg.vocab_size
        print(f"Training with kv of {data_cfg.num_kv_pairs} vocab size {vocab_size} and input seq len {data_cfg.input_seq_len}")
        deltanet_cfg = DeltaNetConfig(
            # core architecture
            attn_mode = "chunk",
            hidden_size = 128,
            expand_k = 1.0,
            expand_v = 1.0,
            use_gate = False,
            use_short_conv = True,
            conv_size = 4,
            use_beta = True,
            use_output_norm = True,
            num_heads = 4,
            qk_norm = 'l2',
            qk_activation = 'silu',
            max_position_embeddings = 2048,
            hidden_ratio = 4,
            intermediate_size = None,
            hidden_act = "swish",
            num_hidden_layers = 1,
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
            vocab_size = vocab_size,
        )
        
        model_cfg = deltanet_cfg
        
        model = DeltaNetForCausalLM(model_cfg).to(torch.device("cuda:0")).to(torch.bfloat16)
        torch.compile(model)

        print(f"Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):.3e}")

        train_dl, test_dl = get_dataloaders(data_cfg)
        optimizer = optim.AdamW(model.parameters(), 
                                lr=1e-4, 
                                weight_decay=1e-6)

        model = train(model, train_dl, optimizer, 30)
