from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from fla.models.transformer.modeling_transformer import TransformerForCausalLM

from model_cfgs.transformer_cfg import cfg as transformer_cfg
from data_gen import DataConfig, get_dataloaders
from hack_utils import non_shifting_loss, compute_accuracy

def train(
    model: TransformerForCausalLM,
    train_dl: DataLoader,
    optimizer: optim.Optimizer,
    max_epochs: int,
    test_dl: DataLoader,
):
    model.train()
    for epoch in range(max_epochs):
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Epoch {epoch} - Model parameters: {num_params:,}")
        for inputs, targets in tqdm(train_dl, desc=f"Train Epoch {epoch}/{max_epochs}"):
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            optimizer.zero_grad()
            output = model(inputs, attention_mask=torch.ones_like(inputs))
            logits = output.logits
            loss = non_shifting_loss(logits, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        # Evaluate at end of epoch
        accuracy = eval(model, test_dl)
        print(f"Epoch {epoch}, Eval Accuracy: {accuracy}")
        model.train()
    return model

def eval(model: TransformerForCausalLM, test_dl: DataLoader):
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_dl:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            output = model(inputs, attention_mask=torch.ones_like(inputs))
            logits = output.logits
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return compute_accuracy(all_logits, all_targets)

if __name__ == "__main__":
    kv = 512 # 16, 32, 64, 128, 256
    data_cfg = DataConfig(
        num_train_examples=100_000,
        num_test_examples=3_000,
        input_seq_len=4*kv,
        vocab_size=4*kv + 1,
        batch_size=1024,
        num_kv_pairs=kv,
        train_power_a=0.01,
        test_power_a=0.01,
        random_non_queries=False,
        seed=37,
    )
    train_dl, test_dl = get_dataloaders(data_cfg)

    model_cfg = transformer_cfg
    model_cfg.vocab_size = data_cfg.vocab_size
    model = TransformerForCausalLM(model_cfg).to(torch.device("cuda:0")).to(torch.bfloat16)
    print(model)
    torch.compile(model)

    optimizer = optim.AdamW(model.parameters(), 
                            lr=5e-4, 
                            weight_decay=1e-6)

    # from ipdb import set_trace; set_trace()
    model = train(model, train_dl, optimizer, 30, test_dl)
    accuracy = eval(model, test_dl)
    print(f"Accuracy: {accuracy}")
    # from ipdb import set_trace; set_trace()