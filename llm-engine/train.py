import torch

from config import GPTConfig
from tokenizer import SimpleTokenizer
from model import GPT


with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()


tokenizer = SimpleTokenizer(text)

tokens = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)

config = GPTConfig(
    vocab_size=len(tokenizer.stoi),
    max_seq_len=4
)

train_size = int(0.9 * len(tokens))

train_data = tokens[:train_size]
val_data = tokens[train_size:]

def get_batch(split):

    data = train_data if split == "train" else val_data

    ix = torch.randint(
        len(data) - config.max_seq_len,
        (config.batch_size,)
    )

    x = torch.stack(
        [
            data[i : i + config.max_seq_len]
            for i in ix
        ]
    )

    y = torch.stack(
        [
            data[i + 1 : i + config.max_seq_len + 1]
            for i in ix
        ]
    )

    return x, y


device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT(config)

model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate
)

x, y = get_batch("train")

x = x.to(device)
y = y.to(device)

logits, loss = model(x, y)

optimizer.zero_grad()

loss.backward()

optimizer.step()

print(loss.item())