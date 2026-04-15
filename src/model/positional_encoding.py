
import torch
import math
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model:int,max_len:int=500,dropout:float=0.1):
        super().__init__()
        self.dropout=torch.nn.Dropout(dropout)

        pe= torch.zeros(max_len,d_model)
        position= torch.arange(0,max_len,dtype=torch.float).unsqueeze(1) #(max_len, 1)

        div_term=torch.exp(
            torch.arange(0,d_model,2,dtype=torch.float)
            * (-math.log(10000.0)/ d_model)
        )
        #use sin for even dims & consin for odd
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe= pe.unsqueeze(0)
        self.register_buffer("pe",pe) #as a part of state

    def forward(self,x:torch.Tensor)->torch.Tensor:
       x = x + self.pe[:, :x.size(1), :]
       return self.dropout(x)


#PositionalEncoding(d_model=256)