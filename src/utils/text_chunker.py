import re
from typing import List,Tuple


class ArabicTextChunker:
    SENTENCE_ENDINGS = re.compile(r'([.؟!،]\s*)')

    def __init__(self,
                 max_chunk_size:int=140,
                 min_chunk_size:int=30,         
                overlap_chars:int=30
                 ):
    
        self.max_chunk_size=max_chunk_size
        self.min_chunk_size=min_chunk_size
        self.overlap_chars=overlap_chars
    

    def chunk(self, text:str)->List[str]:
        if len(text) <=self.max_chunk_size:
            return [text.strip()]
    

        chunks= []
        start=0
        while start < len(text):
            end= start + self.max_chunk_size

            if end>= len(text):
                ch= text[start:].strip()
                if ch:
                    chunks.append(ch)
                break


            segment= text[start:end]
            split_point= self._find_best_split(segment,start)

            if split_point is None:
                last_space= segment.rfind(' ')
                if last_space > self.min_chunk_size:
                    split_point= start + last_space +1
                else:
                    split_point = end


             
            

            ch = text[start:split_point].strip()
            if ch:
                chunks.append(ch)
            
            if self.overlap_chars > 0 and start > 0:
                start= split_point - self.overlap_chars
                if start < 0:
                    start= 0 
            else:
                start = split_point

        return chunks

    
    def _find_best_split(self,segment:str,global_start:int)->int:
        for i in range(len(segment) - 1, self.min_chunk_size - 1, -1):
            if segment[i] in '.؟!،':
                return global_start + i + 1  
        return None
    
  