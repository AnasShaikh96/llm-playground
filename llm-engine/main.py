from tokenizer import SimpleTokenize

text = "Hello World"

tokenizer = SimpleTokenize(text)

encoded = tokenizer.encode(text)

print(encoded)

decoded = tokenizer.decode(encoded)

print(decoded)