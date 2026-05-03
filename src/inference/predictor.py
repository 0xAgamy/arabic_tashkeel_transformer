import torch
import json
from pathlib import Path
import os
import sys
from typing import Optional
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.tokenizer import ArabicCharTokenizer
from .beam_search import beam_search_decode
from .greedy_decode import greedy_decode
from model.transformer import TashkeelTransformer
from utils.text_chunker import ArabicTextChunker

class TashkeelPredictor:
    def __init__(self,
                  checkpoint_path="checkpoints",
                  assets_path="assets",
                    device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load config & tokenizer
        with open(f"{assets_path}/config.json", "r") as f:
            cfg = json.load(f)
        with open(f"{assets_path}/tokenizer.json", "r", encoding="utf-8") as f:
            tok_data = json.load(f)
        
        ## Define the txt chunker
        self.chunker = ArabicTextChunker(
           
          
        )
        
        self.tokenizer = ArabicCharTokenizer()
        self.tokenizer.char2id = {k: int(v) for k, v in tok_data["char2id"].items()}
        self.tokenizer.id2char = {int(k): v for k, v in tok_data["id2char"].items()}
        self.max_src_len = cfg["max_src_len"]
        
        # Initialize model with saved hyperparameters
        self.model = TashkeelTransformer(
            src_vocab_size=cfg["vocab_size"], tgt_vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"], n_heads=8,
            num_encoder_layers=cfg["num_encoder_layers"],
            num_decoder_layers=cfg["num_decoder_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"], pad_idx=ArabicCharTokenizer.PAD
        ).to(self.device)
        
        # Load weights
        ckpt = torch.load(f"{checkpoint_path}/best_model.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print(f"Model loaded on {self.device}")


    def _diacritize_single(self, text: str,use_beam:Optional[bool]=False, beam_size: int = 4)->str:
        src_ids = self.tokenizer.encode(text, add_sos=False, add_eos=True)
        max_len = self.max_src_len
        if len(src_ids) > max_len:
            src_ids = src_ids[:max_len]
        
        with torch.no_grad():
            if use_beam:
                return beam_search_decode(
                    self.model, src_ids, self.tokenizer, self.device, beam_size=beam_size
                )
            return greedy_decode(
                self.model, src_ids, self.tokenizer, self.device,
            )
           


    def diacritize(self, text: str, use_beam:Optional[bool]=False, beam_size: int = 4) -> str:
        """Diacritize a single Arabic sentence."""
        if len(text.strip()) == 0: return ""
        
        if len(text.strip()) <= self.chunker.max_chunk_size:
            return self._diacritize_single(text.strip(),use_beam, beam_size)

        chunks= self.chunker.chunk(text.strip())
        diacritized_chunks = [
            self._diacritize_single(chunk,use_beam,  beam_size)
            for chunk in chunks
        ]
        return " ".join(diacritized_chunks)


    # def diacritize_batch(self, texts: list, use_beam: bool = True, beam_size: int = 4) -> list:
    #     """Process multiple sentences efficiently."""
    #     return [self.diacritize(t, use_beam, beam_size) for t in texts]
    