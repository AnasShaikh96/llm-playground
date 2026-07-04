import os

import torch

from config import GPTConfig
from tokenizer import SimpleTokenizer
from model import GPT


CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model.pt")
TOKENIZER_PATH = os.path.join(CHECKPOINT_DIR, "tokenizer.json")


os.makedirs(CHECKPOINT_DIR, exist_ok=True)


with open("./data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()


tokenizer = SimpleTokenizer.build(text)
tokenizer.save(TOKENIZER_PATH)


tokens = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long,
)

config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    max_seq_len=32,
)

train_size = int(0.9 * len(tokens))

train_data = tokens[:train_size]
val_data = tokens[train_size:]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device",device)

def get_batch(split):

    data = train_data if split == "train" else val_data

    ix = torch.randint(
        len(data) - config.max_seq_len,
        (config.batch_size,),
    )

    x = torch.stack(
        [
            data[i:i + config.max_seq_len]
            for i in ix
        ]
    )

    y = torch.stack(
        [
            data[i + 1:i + config.max_seq_len + 1]
            for i in ix
        ]
    )

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():

    model.eval()

    losses = {}

    for split in ["train", "val"]:

        values = []

        for _ in range(20):

            x, y = get_batch(split)

            _, loss = model(x, y)

            values.append(loss.item())

        losses[split] = sum(values) / len(values)

    model.train()

    return losses


model = GPT(config).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
)


for step in range(config.max_iters):

    if step % config.eval_interval == 0:

        losses = estimate_loss()

        print(
            f"Step {step} | "
            f"Train {losses['train']:.4f} | "
            f"Val {losses['val']:.4f}"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
            },
            MODEL_PATH,
        )

    x, y = get_batch("train")

    _, loss = model(x, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
    },
    MODEL_PATH,
)

print("Training complete.")