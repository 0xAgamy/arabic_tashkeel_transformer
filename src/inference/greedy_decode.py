
import torch
import torch.nn.functional as F
from typing import List
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.transformer import TashkeelTransformer
from data.tokenizer import ArabicCharTokenizer

MAX_TGT_LEN=500
@torch.no_grad()
def greedy_decode(model:TashkeelTransformer,
                       src_ids:List[int],
                       tokenizer:ArabicCharTokenizer,
                       device=torch.device,
                      
                       max_len:int=MAX_TGT_LEN)->str: 

    model.eval()

    src=torch.tensor([src_ids], dtype=torch.long, device=device) 
    memory= model.encode(src)
    

    generated= [tokenizer.SOS]
    
    for _ in range(max_len):
        tgt= torch.tensor([generated], dtype=torch.long,device=device)
        tgt_len= tgt.size(1)

        tgt_mask= model.make_causal_mask(tgt_len,device)

        decoder_out= model.decoder(tgt,memory,tgt_mask=tgt_mask)
        logits= model.output_projection(decoder_out[:,-1, :])

        next_token= logits.argmax(dim=-1).item()
        generated.append(next_token)

        if next_token== tokenizer.EOS: break
    
    return tokenizer.decode(generated)