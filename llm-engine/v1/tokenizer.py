import json


class SimpleTokenizer:

    def __init__(self, stoi=None):

        self.stoi = stoi or {}

        self.itos = {
            value: key
            for key, value in self.stoi.items()
        }

    @classmethod
    def build(cls, text):

        chars = sorted(set(text))

        stoi = {
            ch: idx
            for idx, ch in enumerate(chars)
        }

        return cls(stoi)

    def encode(self, text):

        return [
            self.stoi[ch]
            for ch in text
        ]

    def decode(self, tokens):

        return "".join(
            self.itos[token]
            for token in tokens
        )

    def save(self, path):

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f)

    @classmethod
    def load(cls, path):

        with open(path, "r", encoding="utf-8") as f:
            stoi = json.load(f)

        return cls(stoi)

    @property
    def vocab_size(self):

        return len(self.stoi)