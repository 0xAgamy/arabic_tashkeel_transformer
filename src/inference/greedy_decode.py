
import torch
import torch.nn.functional as F
from typing import List
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.transformer import TashkeelTransformer
from data.tokenizer import ArabicCharTokenizer

MAX_TGT_LEN=500
@torch.no_grad()
def greedy_decode(
    model: TashkeelTransformer,
    src_ids: List[int],
    tokenizer: ArabicCharTokenizer,
    device: torch.device,
    max_len: int = 500
) -> str:
    model.eval()
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    memory = model.encode(src)
    
   
    max_mask = model.make_causal_mask(max_len, device)
    
    
    generated = torch.full((1, 1), tokenizer.SOS, dtype=torch.long, device=device)
    
    for step in range(max_len - 1):
        tgt_len = generated.size(1)
        
       
        tgt_mask = max_mask[:tgt_len, :tgt_len]
        
        decoder_out = model.decoder(generated, memory, tgt_mask=tgt_mask)
        logits = model.output_projection(decoder_out[:, -1, :])
        
        
        next_token = logits.argmax(dim=-1, keepdim=True) 
        generated = torch.cat([generated, next_token], dim=1)
        
        if next_token.item() == tokenizer.EOS:
            break
            
    return tokenizer.decode(generated[0].tolist())