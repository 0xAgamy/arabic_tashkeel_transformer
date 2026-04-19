from torch.utils.data import Dataset, DataLoader
import torch
from typing import List
from .tokenizer import ArabicCharTokenizer
MAX_SRC_LEN=150 #max undiacritized chars per sample
MAX_TGT_LEN=250 #  max diacritized chars (includes harakat between chars)


class TashkeelDataset(Dataset):
    """Handle batching, shuffling and multiworker.

    Return:
        src_ids — encoded undiacritized source, padded to MAX_SRC_LEN
        tgt_ids — encoded diacritized target (WITH <SOS> prepended), padded
        tgt_labels — same as tgt_ids but SHIFTED LEFT by 1 (for teacher forcing)
    """
    def __init__(self,sources:List[str],
                 targets:List[str],
                 tokenizer:ArabicCharTokenizer,
                 max_src_len:int=MAX_SRC_LEN,
                 max_tgt_len:int=MAX_TGT_LEN):
        super().__init__()
        self.tokenizer=tokenizer
        self.max_src_len=max_src_len
        self.max_tgt_len=max_tgt_len

        self.pairs=[]
        for src,tgt in zip(sources,targets):
            src_ids=tokenizer.encode(src,add_sos=False,add_eos=True)
            # tgt_ids= tokenizer.encode(tgt,add_sos=True,add_eos=True)
            # #Labels = target shifted left 
            # tgt_labels=tokenizer.encode(tgt,add_sos=False,add_eos=True)
            tgt_full = tokenizer.encode(tgt, add_sos=False, add_eos=True)
            tgt_ids = [tokenizer.SOS] + tgt_full[:-1]
            # Labels = target shifted left (no SOS, but EOS included)
            tgt_labels = tgt_full
            #Truncate
            src_ids=src_ids[:max_src_len]
            tgt_ids=tgt_ids[:max_tgt_len]
            tgt_labels = tgt_labels[:max_tgt_len]

            self.pairs.append(
                (src_ids,tgt_ids,tgt_labels)
            )
    

    def __len__(self)->int:
        return len(self.pairs)
    
    def __getitem__(self,idx:int):

        src_ids,tgt_ids,tgt_lables= self.pairs[idx]
        return{
            "src_ids":torch.tensor(src_ids,dtype=torch.long),
            "tgt_ids":torch.tensor(tgt_ids,dtype=torch.long),
            "tgt_labels":torch.tensor(tgt_lables,dtype=torch.long)
        }
    



def collate_fn(batch:List[dict])->dict:
    """pad shorter ones,
    to the length of the longest in the batch (dynamic padding)."""

    PAD= ArabicCharTokenizer.PAD

    src_padded=torch.nn.utils.rnn.pad_sequence(
        [item['src_ids'] for item in batch],
        batch_first=True,padding_value=PAD
    )

    tgt_padded= torch.nn.utils.rnn.pad_sequence(
        [item['tgt_ids'] for item in batch],
        batch_first=True, padding_value=PAD
    )

    lbl_padded= torch.nn.utils.rnn.pad_sequence(
        [item['tgt_labels'] for item in batch],
        batch_first=True,padding_value=PAD
    )

    #padding mask
    src_key_padding_mask= (src_padded==PAD) 
    tgt_key_padding_mask= (tgt_padded==PAD)

    return{
        "src":src_padded,
        "tgt":tgt_padded,
        "labels":lbl_padded,
        "src_key_padding_mask":src_key_padding_mask,
        "tgt_key_padding_mask":tgt_key_padding_mask
        }