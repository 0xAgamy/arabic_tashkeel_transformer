import torch
import torch.nn.functional as F
from typing import List, Tuple
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.transformer import TashkeelTransformer
from data.tokenizer import ArabicCharTokenizer
MAX_TGT_LEN=250
@torch.no_grad()
def beam_search_decode(
    model: TashkeelTransformer,
    src_ids: List[int],
    tokenizer: ArabicCharTokenizer,
    device: torch.device,
    beam_size: int = 4,
    max_len: int = MAX_TGT_LEN,
    length_penalty: float = 0.6,
) -> str:
   
    model.eval()
 
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    memory = model.encode(src)  # (1, src_len, d_model)
 
    # Each beam: (sequence of token IDs, cumulative log probability)
    beams: List[Tuple[List[int], float]] = [([tokenizer.SOS], 0.0)]
    completed: List[Tuple[List[int], float]] = []
 
    for step in range(max_len):
        if not beams:
            break
 
        candidates: List[Tuple[List[int], float]] = []
 
        for seq, score in beams:
            tgt = torch.tensor([seq], dtype=torch.long, device=device)
            tgt_mask = model.make_causal_mask(len(seq), device)
 
            decoder_out = model.decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = model.output_projection(decoder_out[:, -1, :])  
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)    
 
         
            topk_log_probs, topk_ids = log_probs.topk(beam_size)
            for log_p, tok_id in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                new_seq = seq + [tok_id]
                new_score = score + log_p
 
                if tok_id == tokenizer.EOS:
                   
                    normalized = new_score / (len(new_seq) ** length_penalty)
                    completed.append((new_seq, normalized))
                else:
                    candidates.append((new_seq, new_score))
 

        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_size]

    if not completed:
        completed = [(seq, score / (len(seq) ** length_penalty)) for seq, score in beams]
 

    best_seq = max(completed, key=lambda x: x[1])[0]
    return tokenizer.decode(best_seq)