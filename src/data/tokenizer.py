#   Special tokens:
#       <PAD> = 0   — padding to batch variable-length sequences
#       <SOS> = 1   — start-of-sequence (decoder prompt)
#       <EOS> = 2   — end-of-sequence (decoder stop signal)
#       <UNK> = 3   — unknown character (fallback)
from typing import Dict, List
class ArabicCharTokenizer:
    PAD, SOS, EOS, UNK = 0, 1, 2, 3
    SPECIAL = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
    
    def __init__(self):
        self.char2id: Dict[str,int]={}
        self.id2char: Dict[int,str]={}
        for i ,tok in enumerate(self.SPECIAL):
            self.char2id[tok]= i
            self.id2char[i]=tok

    def build_vocab(self, texts:List[str])->None:
        chars= set()
        for text in texts:
            chars.update(text)
        chars= sorted(chars) # deterministic order

        start_id= len(self.SPECIAL)
        for i, ch in enumerate(chars):
            self.char2id[ch] = start_id +i
            self.id2char[start_id+i]= ch
        print(f"[Tokenizer] Vocabulary size: {len(self.char2id)}") 
        
    
    def encode(self, text:str,add_sos:bool=False, add_eos:bool=True)-> List[int]:
        ids= []
        if add_sos:
            ids.append(self.SOS)
        for ch in text:
            ids.append(self.char2id.get(ch,self.UNK))
        if add_eos:
            ids.append(self.EOS)
        return ids
    def decode(self, ids:List[int], skip_speicals:bool=True)->str:
        chars=[]
        for i in ids:
            if skip_speicals and i in (self.PAD,self.SOS,self.EOS,self.UNK):
                continue
            chars.append(self.id2char.get(i,"?"))
        return "".join(chars)
   
    @property
    def vocab_size(self) -> int:
        return len(self.char2id)
    @property
    def vocabs(self)->Dict[str,int]:
        return self.char2id