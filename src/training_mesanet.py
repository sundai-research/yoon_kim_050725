import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from fla.models.mesa_net.modeling_mesa_net import MesaNetForCausalLM
from fla.models.mesa_net.configuration_mesa_net import MesaNetConfig

from data_gen import DataConfig, get_dataloaders
from hack_utils import non_shifting_loss, compute_accuracy

def train(
    model: MesaNetForCausalLM,
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

def eval(model: MesaNetForCausalLM, test_dl: DataLoader):
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


# === Rohan ===
def get_vocab_size(kv: int):
    return 4 * kv + 1


def get_data_config(kv: int):
    return DataConfig(
        num_train_examples=100_000,
        num_test_examples=3_000,
        input_seq_len=4 * kv,
        vocab_size=get_vocab_size(kv),
        batch_size=256,
        num_kv_pairs=kv,
        train_power_a=0.01,
        test_power_a=0.01,
        random_non_queries=False,
        seed=37,
        cache_path=f"data_{kv}.pt"
    )

def get_model_config(kv: int):
    return MesaNetConfig(
        vocab_size=get_vocab_size(kv),
        hidden_size=256,
        num_hidden_layers=2,
        num_heads=1,
        conv_size=4,
    )



if __name__ == "__main__":
    kv_options = (16, 64, 256, 512)  # three settings
    for kv in kv_options:
        data_cfg = get_data_config(kv)
        train_dl, test_dl = get_dataloaders(data_cfg)

        model_cfg = get_model_config(kv)
        model = MesaNetForCausalLM(model_cfg).to(torch.device("cuda:0")).to(torch.bfloat16)
        torch.compile(model)

        optimizer = optim.AdamW(model.parameters(), 
                                lr=1e-4, 
                                weight_decay=1e-6)

        # from ipdb import set_trace; set_trace()
        model = train(model, train_dl, optimizer, max_epochs=10)
        accuracy = eval(model, test_dl)
        print(f"kv={data_cfg.num_kv_pairs}, input_seq_len={data_cfg.input_seq_len}, vocab_size={data_cfg.vocab_size}")
        print(f"Accuracy: {accuracy}")
        # from ipdb import set_trace; set_trace()