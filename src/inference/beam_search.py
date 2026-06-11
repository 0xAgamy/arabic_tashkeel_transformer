import torch
import torch.nn.functional as F
from typing import List, Tuple
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.transformer import TashkeelTransformer
from data.tokenizer import ArabicCharTokenizer
MAX_TGT_LEN=500
@torch.no_grad()
def beam_search_decode(
    model: TashkeelTransformer,
    src_ids: List[int],
    tokenizer: ArabicCharTokenizer,
    device: torch.device,
    beam_size: int = 4,
    max_len: int = 500,
    length_penalty: float = 0.6,
) -> str:
    model.eval()
    
  
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    memory = model.encode(src)
    
    
    max_mask = model.make_causal_mask(max_len, device)
    
    
    sequences = torch.full((1, 1), tokenizer.SOS, dtype=torch.long, device=device)
    scores = torch.zeros(1, device=device)
    
    completed_sequences = []
    completed_scores = []
    
    for step in range(max_len - 1):
        current_beam_size = sequences.size(0)
        current_len = sequences.size(1)
        
        
        mem_expanded = memory.expand(current_beam_size, -1, -1).contiguous()
        
       
        tgt_mask = max_mask[:current_len, :current_len]
        
        
        decoder_out = model.decoder(sequences, mem_expanded, tgt_mask=tgt_mask)
        logits = model.output_projection(decoder_out[:, -1, :]) )
        log_probs = F.log_softmax(logits, dim=-1)
        
       
        next_scores = scores.unsqueeze(1) + log_probs 
        next_scores_flat = next_scores.view(-1)
        
        k = min(beam_size, next_scores_flat.size(0))
        topk_scores, topk_ids = torch.topk(next_scores_flat, k)
        
        beam_ids = topk_ids // tokenizer.vocab_size
        token_ids = topk_ids % tokenizer.vocab_size
        
        sequences = sequences[beam_ids]
        sequences = torch.cat([sequences, token_ids.unsqueeze(1)], dim=1)
        scores = topk_scores
        
        is_eos = (token_ids == tokenizer.EOS)
        
        if is_eos.any():
            for i in range(k):
                if is_eos[i]:
                    seq = sequences[i].tolist()
                    score = scores[i].item()
                    norm_score = score / (len(seq) ** length_penalty)
                    completed_sequences.append(seq)
                    completed_scores.append(norm_score)
            
            active_mask = ~is_eos
            sequences = sequences[active_mask]
            scores = scores[active_mask]
            
        if sequences.size(0) == 0:
            break
            
    if not completed_sequences:
        best_idx = torch.argmax(scores).item()
        best_seq = sequences[best_idx].tolist()
        best_score = scores[best_idx].item()
        norm_score = best_score / (len(best_seq) ** length_penalty)
        completed_sequences.append(best_seq)
        completed_scores.append(norm_score)
        
    best_overall_idx = max(range(len(completed_scores)), key=lambda i: completed_scores[i])
    best_seq = completed_sequences[best_overall_idx]
    
    return tokenizer.decode(best_seq)