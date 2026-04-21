## Set imports
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.tokenizer import ArabicCharTokenizer
from torch.utils.data import DataLoader
from src.data.dataset import TashkeelDataset, collate_fn
from src.model.transformer import TashkeelTransformer
from src.training.trainer import train

## Default 
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE='cpu'
BATCH_SIZE = 32
D_MODEL = 256
NUM_EPOCHS = 50
TRAIN_SPLIT = 0.9


## Get data

import json
with open("data/processed/data.json", 'r')as f:
    data= json.load(f)

sources = [s for item in data for s in item["src"]]
targets = [t for item in data for t in item["tgt"]]

print(f"Loaded {len(sources)} sentence pairs.")
print(f"Example:\n  SRC: {sources[0]}\n  TGT: {targets[0]}")


## Get Tokenizer

tokenizer = ArabicCharTokenizer()
tokenizer.build_vocab(sources + targets)  # build from BOTH sides
print(f"Vocab size: {tokenizer.vocab_size}")


## split dataset


n = len(sources)
split = int(n * TRAIN_SPLIT)
train_src, train_tgt = sources[:split], targets[:split]
val_src, val_tgt = sources[split:], targets[split:]

train_dataset = TashkeelDataset(train_src, train_tgt, tokenizer)
val_dataset = TashkeelDataset(val_src, val_tgt, tokenizer)

train_loader = DataLoader(
  train_dataset, batch_size=BATCH_SIZE, shuffle=True,
  collate_fn=collate_fn, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
  val_dataset, batch_size=BATCH_SIZE, shuffle=False,
  collate_fn=collate_fn, num_workers=2, pin_memory=True
)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")


## Build model

model = TashkeelTransformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        pad_idx=ArabicCharTokenizer.PAD,
    ).to(DEVICE)
 
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")

print("Start Learning")
train(model, train_loader, val_loader, DEVICE, num_epochs=NUM_EPOCHS)