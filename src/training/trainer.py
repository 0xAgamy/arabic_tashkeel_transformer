
from torch.utils.data import DataLoader
import torch
import os
import sys
from .scheduler import WarmupCosineScheduler
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.transformer import TashkeelTransformer
from src.data.tokenizer import ArabicCharTokenizer

def train_epoch(model: TashkeelTransformer,
                dataloader:DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler,
                criterion:torch.nn.CrossEntropyLoss,
                 device:torch.device,
                  clip_grad_norm:float = 1.0 )->float:
    
    model.train()
    total_loss=0.0
    n_batches=0

    for batch in dataloader:
        src= batch['src'].to(device)
        tgt= batch['tgt'].to(device)
        labels= batch['labels'].to(device)
        src_mask = batch["src_key_padding_mask"].to(device)
        tgt_mask = batch["tgt_key_padding_mask"].to(device)

        logits= model(src,tgt,src_key_padding_mask=src_mask)

        B, T,V= logits.shape

        loss= criterion(logits.reshape(B * T, V) ,labels.reshape(B*T))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),clip_grad_norm)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        n_batches +=1
    return total_loss / max(n_batches, 1)


def evaluate(model: TashkeelTransformer,
             dataloader: DataLoader,
             criterion: torch.nn.CrossEntropyLoss,
             device:torch.device)-> float:
    model.eval()
    total_loss= 0.0
    n_batches=0

    for batch in dataloader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        labels = batch["labels"].to(device)
        src_mask = batch["src_key_padding_mask"].to(device)
 
        logits = model(src, tgt, src_key_padding_mask=src_mask)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), labels.reshape(B * T))
 
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)

def train(model:TashkeelTransformer,
          train_loader:DataLoader,
          val_loader:DataLoader,
          device: torch.device,
          num_epochs:int =20,
          learning_rate:float= 1e-4,
          checkpoint_dir:str="checkpoints"):
    os.makedirs(checkpoint_dir,exist_ok=True)
    history=[]
    optimizer= torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),   #Transformer paper values
        eps=1e-9,
        weight_decay=0.01
    )

    scheduler= WarmupCosineScheduler(optimizer=optimizer,d_model=model.d_model)
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=ArabicCharTokenizer.PAD,
        label_smoothing=0.1,
    )

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs +1):
        train_loss= train_epoch(model,train_loader,optimizer,scheduler,criterion,device)
        val_loss= evaluate(model,val_loader,criterion,device)

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss= val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                os.path.join(checkpoint_dir, "best_model.pt"),
            )
            print(f"Saved best model (val_loss={val_loss:.4f})")

        history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        })
        pd.DataFrame(history).to_csv(f"{checkpoint_dir}/training_log.csv", index=False)

