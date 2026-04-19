import torch
import math
from typing import Optional
from .positional_encoding import PositionalEncoding
class TashkeelTransformer(torch.nn.Module):
    def __init__(self,
                src_vocab_size:int,
                tgt_vocab_size:int,
                d_model:int=256,
                n_heads:int=8,
                num_encoder_layers:int=3,
                num_decoder_layers:int=3,
                dim_feedforward:int=512,
                dropout:float=0.1,
                max_seq_len:int=500,
                pad_idx:int=0
                ):
        super().__init__()
        self.d_model= d_model
        self.pad_idx= pad_idx
        self.tgt_vocab_size= tgt_vocab_size

        # Embedding

        self.src_embedding= torch.nn.Embedding(src_vocab_size,d_model,padding_idx=pad_idx)
        self.tgt_embedding= torch.nn.Embedding(tgt_vocab_size,d_model,padding_idx=pad_idx)
        self.scale= math.sqrt(d_model) 
        
        self.pos_encoding= PositionalEncoding(d_model,max_seq_len,dropout)


        # Transformer

        self.transformer= torch.nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.output_projection= torch.nn.Linear(d_model,tgt_vocab_size)
        self.output_projection.weight = self.tgt_embedding.weight
        self._init_weights()


    def _init_weights(self):
        for p in self.parameters():
            if p.dim() >1 :
                torch.nn.init.xavier_uniform_(p)
        torch.nn.init.normal_(self.src_embedding.weight,mean=0, std=0.01)
        torch.nn.init.normal_(self.tgt_embedding.weight,mean=0,std=0.01)

    @staticmethod
    def make_causal_mask(sz:int,device:torch.device) -> torch.Tensor:
        mask= torch.triu(torch.ones(sz,sz,device=device),diagonal=1)
        return mask.masked_fill(mask==1 ,float("-inf"))

    def encode(self,
               src:torch.Tensor,
               src_key_padding_mask:Optional[torch.Tensor]=None)->torch.Tensor:
        src_emb= self.pos_encoding(self.src_embedding(src) * self.scale)
        return self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask)
    

    def decoder(self,
                tgt:torch.Tensor,
                memory:torch.Tensor,
                tgt_mask:Optional[torch.Tensor]=None,
                tgt_key_padding_mask:Optional[torch.Tensor]=None,
                memory_key_padding_mask:Optional[torch.Tensor]=None)->torch.Tensor:
        tgt_emb= self.pos_encoding(self.tgt_embedding(tgt) * self.scale)

        return self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )   
    

    def forward(self,
                src:torch.Tensor,
                tgt:torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                )-> torch.Tensor:
        tgt_len = tgt.size(1)
        device = src.device

        tgt_mask = self.make_causal_mask(tgt_len, device)
        memory = self.encode(src, src_key_padding_mask)

        decoder_out = self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        logits = self.output_projection(decoder_out) 
        return logits
