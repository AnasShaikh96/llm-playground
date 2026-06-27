class SimpleTokenize: 
    def __init__(self, text):
        chars = sorted(list(set(text)))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i:ch for i, ch in enumerate(chars)}

    def encode(self, text):
        return [self.stoi[ch] for ch in text]


    def decode(self, tokens):
        return "".join(self.itos[token] for token in tokens)