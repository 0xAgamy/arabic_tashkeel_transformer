import torch
import torch.nn.functional as F
from typing import List
from src.model.transformer import TashkeelTransformer
from src.data.tokenizer import ArabicCharTokenizer
MAX_TGT_LEN=250
@torch.no_grad()
def beam_search_decode(model:TashkeelTransformer,
                       src_ids:List[int],
                       tokenizer:ArabicCharTokenizer,
                       device=torch.device,
                       beam_search:int=4,
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