import torch

from config import GPTConfig
from tokenizer import SimpleTokenizer
from model import GPT

MODEL_PATH = "./checkpoints/model.pt"
TOKENIZER_PATH = "./checkpoints/tokenizer.json"

device = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = SimpleTokenizer.load(TOKENIZER_PATH)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
)

config = GPTConfig(
    **checkpoint["config"]
)

model = GPT(config)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)

prompt = input("Prompt: ")

tokens = tokenizer.encode(prompt)

x = torch.tensor(
    [tokens],
    dtype=torch.long,
    device=device,
)

generated = model.generate(
    x,
    max_new_tokens=200,
)

print(
    tokenizer.decode(
        generated[0].tolist()
    )
)