import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.tokenizer import ArabicCharTokenizer
import json

with open("data/processed/data.json", "r") as f:
    data=json.load(f)

all_texts = []
for item in data:
    all_texts.extend(item["src"])
    all_texts.extend(item["tgt"])

print(len(all_texts))
tok=ArabicCharTokenizer()
tok.build_vocab(data[0]['src'])
ids= tok.encode("الْعِلْمِ")
print(f"Encodeing: {ids} ")

chars= tok.decode(ids)
print(f"Decoded : {chars} ")
vocab_size= tok.vocab_size
vocabs=tok.vocabs
print(f"Vocab size : {vocab_size}\nVocabs: {vocabs}")