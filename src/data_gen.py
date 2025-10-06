from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import hashlib
import pickle
import os
from pathlib import Path

@dataclass
class Data:
    train_inputs: torch.Tensor
    train_labels: torch.Tensor
    test_inputs: torch.Tensor
    test_labels: torch.Tensor

@dataclass
class DataConfig:
    num_train_examples: int = 10_000
    num_test_examples: int = 1000
    input_seq_len: int = 64
    vocab_size: int = 65
    batch_size: int = 32
    num_kv_pairs: int = 16
    train_power_a: float=0.01
    test_power_a: float=0.01
    random_non_queries: bool=False,
    seed: int=0

def multiquery_ar(
    vocab_size: int=8_192,
    num_train_examples: int=100_000,
    num_test_examples: int=3_000,
    input_seq_len: int=64,
    num_kv_pairs: int=4,
    train_power_a: float=0.01,
    test_power_a: float=0.01,
    random_non_queries: bool=True,
    seed: int=0,
    use_cache: bool=True,
) -> Data:
    """
    Generate synthetic sequences for the multi-query associative recall (MQAR) task.

    Each example is an integer sequence of length `input_seq_len` over
    `[0, vocab_size - 1]`. The sequence is composed of:
    - Context prefix of length `2 * num_kv_pairs`: alternating `key, value, key, value, ...`.
      Keys are sampled with replacement from `{1, ..., vocab_size // 2 - 1}` and values
      from `{vocab_size // 2, ..., vocab_size - 1}`. Token `0` is reserved for blanks.
    - A suffix that contains `num_kv_pairs` query key tokens interleaved with filler tokens.
      Query locations are chosen without replacement using a power-law gap distribution
      parameterized by `power_a` (see note below). By default filler zeros can be
      randomized to value tokens when `random_non_queries=True`.

    Targets are computed online from the input. At every position that contains a key `k`,
    the label is the most recently observed value assigned to `k` in the preceding prefix;
    all other positions are set to `-100` (ignored by loss). Consequently, labels are only
    meaningful at key positions (including query keys in the suffix).

    Power-law gap distribution note:
    Let `space = (input_seq_len - 2 * num_kv_pairs) // 2`. Query gaps are sampled from
    `{1, ..., space}` with probabilities proportional to `(i) ** (power_a - 1)`.
    Smaller `power_a` concentrates queries closer to the context; `power_a = 1.0`
    yields a uniform distribution. Example visualization:
    ```
    space = 100
    power_a = 0.01
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()
    ```

    Example:
        `multiquery_ar(vocab_size=12, num_kv_pairs=2, input_seq_len=16, random_non_queries=False)`
        might produce:
            Inputs: 2 8 4 7 0 0 4 0 0 0 0 0 2 0 0 0
            Labels: -100 -100 -100 -100 -100 -100 7 -100 -100 -100 -100 -100 8 -100 -100 -100
        Setting `random_non_queries=True` replaces the zeros with random values from the
        value vocabulary but leaves labels unchanged.

    Constraints:
    - `input_seq_len` must be even.
    - `input_seq_len >= 4 * num_kv_pairs` (space must allow placing `num_kv_pairs` queries).
    - `vocab_size > input_seq_len`.

    Args:
        vocab_size (int): Vocabulary size. Large vocabularies (e.g., > 1k) can stress
            recall more strongly. Default: 8192.
        num_train_examples (int): Number of training examples to generate. Default: 100_000.
        num_test_examples (int): Number of test examples to generate. Default: 3_000.
        input_seq_len (int): Total input sequence length. Must be even. Default: 64.
        num_kv_pairs (int): Number of `(key, value)` pairs in the context and number of queries.
            Default: 4.
        train_power_a (float): Power-law parameter for training set query spacing.
            `1.0` ≈ uniform; smaller values bias toward shorter gaps. Default: 0.01.
        test_power_a (float): Power-law parameter for test set query spacing. Default: 0.01.
        random_non_queries (bool): If True, replace filler zeros with random values sampled
            from the value vocabulary. Default: True.
        seed (int): Base RNG seed. The test set uses `seed + 10`. Default: 0.
        use_cache (bool): If True, cache generated data to /tmp and reuse on subsequent calls
            with identical arguments. Default: True.

    Returns:
        Data: Dataclass with fields:
            - `train_inputs`: (num_train_examples, input_seq_len) int64 tensor
            - `train_labels`: (num_train_examples, input_seq_len) int64 tensor with `-100` for non-targets
            - `test_inputs`: (num_test_examples, input_seq_len) int64 tensor
            - `test_labels`: (num_test_examples, input_seq_len) int64 tensor with `-100` for non-targets

    Notes:
    - A simple overlap check prints a warning if >0.1% of test sequences also appear in the
      training set.
    """
    
    # Create cache key from all arguments
    if use_cache:
        cache_key = {
            'vocab_size': vocab_size,
            'num_train_examples': num_train_examples,
            'num_test_examples': num_test_examples,
            'input_seq_len': input_seq_len,
            'num_kv_pairs': num_kv_pairs,
            'train_power_a': train_power_a,
            'test_power_a': test_power_a,
            'random_non_queries': random_non_queries,
            'seed': seed
        }
        
        # Create hash of arguments
        cache_str = str(sorted(cache_key.items()))
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()
        cache_path = Path(f"/tmp/mqar_cache_{cache_hash}.pkl")
        
        # Check if cache exists
        if cache_path.exists():
            print(f"Loading cached data from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

    train_inputs, train_labels = _mqar(
        vocab_size=vocab_size,
        num_examples=num_train_examples,
        input_seq_len=input_seq_len,
        seed=seed,
        power_a=train_power_a,
        num_kv_pairs=num_kv_pairs,
        random_non_queries=random_non_queries
    )
    test_inputs, test_labels = _mqar(
        vocab_size=vocab_size,
        num_examples=num_test_examples,
        input_seq_len=input_seq_len,
        seed=seed + 10,  # different seed for test set
        power_a=test_power_a,
        num_kv_pairs=num_kv_pairs,
        random_non_queries=random_non_queries
    )

    # data = SyntheticData(
    #     train_inputs=train_inputs,
    #     train_labels=train_labels,
    #     test_inputs=test_inputs,
    #     test_labels=test_labels,
    # )

    # check for data leakage:
    train_set = set([" ".join(map(str, x)) for x in train_inputs.tolist()])
    test_set = set([" ".join(map(str, x)) for x in test_inputs.tolist()])
    frac_test_in_train = 1 - (len(test_set - train_set) / len(test_set))
    if frac_test_in_train > 0.001:
        print(
            "WARNING: Potential data leakage detected. " 
            f"{frac_test_in_train: 0.2f} of test examples are in the train set."
        )
    data = Data(
        train_inputs=train_inputs,
        train_labels=train_labels,
        test_inputs=test_inputs,
        test_labels=test_labels,
    )
    
    # Save to cache if caching is enabled
    if use_cache:
        print(f"Saving data to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    return data


def _mqar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float=0.01,
    num_kv_pairs: int=8,
    random_non_queries: bool=True
):
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len

    np.random.seed(seed)

    # two tokens for key and value
    context_size = num_kv_pairs * 2

    # create keys so that each key is present exactly once in each example
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=True, size=num_kv_pairs)

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=True, size=num_kv_pairs)
    # create sequences
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values
    # compute power law
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a-1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)
    # queries and answers
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(
        queries, (gaps * 2),
        values=np.apply_along_axis(np.random.choice, 1, keys, replace=True, size=keys[0].shape),
        axis=1
    )
    examples = np.concatenate([kvs, queries], axis=1)
    inputs = torch.tensor(examples[:, :-1])

    if random_non_queries:
        inputs[inputs == 0] = torch.tensor(
            np.random.choice(value_choices, replace=True, size=inputs[inputs == 0].shape))

    def process_sequence(seq):
        out = np.full((seq.shape[0],), -100, dtype=np.int64)
        state = {}
        curr_key = None
        for i in range(seq.shape[0]):
            if seq[i] in key_choices:
                curr_key = seq[i]
                if curr_key in state:
                    out[i] = state[curr_key]
            elif curr_key is not None:
                state[curr_key] = seq[i]
                curr_key = None
        return out
    labels = torch.tensor(
        np.apply_along_axis(process_sequence, axis=1, arr=inputs))
    # labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    # labels = np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)
    # inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])
    # replace all the 0 with random values
    return inputs, labels

def get_dataloaders(cfg: DataConfig):
    data = multiquery_ar(
        cfg.vocab_size,
        cfg.num_train_examples,
        cfg.num_test_examples,
        cfg.input_seq_len,
        cfg.num_kv_pairs,
        cfg.train_power_a,
        cfg.test_power_a,
        cfg.random_non_queries,
        cfg.seed
    )
    train_dl = DataLoader(
        TensorDataset(data.train_inputs, data.train_labels),
        batch_size=cfg.batch_size, 
        shuffle=True)
    test_dl = DataLoader(
        TensorDataset(data.test_inputs, data.test_labels),
        batch_size=cfg.batch_size, 
        shuffle=False)
    return train_dl, test_dl
