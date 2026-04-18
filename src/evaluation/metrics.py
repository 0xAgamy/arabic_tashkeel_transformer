from typing import List, Tuple, Dict
ARABIC_LETTERS = set(chr(c) for c in range(0x0621, 0x064B))
HARAKAT = {
    "\u064E",   
    "\u064F",   
    "\u0650",   
    "\u0651",  
    "\u0652",   
    "\u064B",   
    "\u064C",   
    "\u064D",   
    "\u0670",   
}

def extract_char_diacritic_pairs(text:str)-> List[Tuple[str,str]]:
    """Extract (base_char, diacritic) pairs from diacritized Arabic text.

    Returns:
        List of (base_char, diacritic_or_empty_string) tuples.
    """
    pairs= []
    i=0
    while i < len(text):
        ch= text[i]
        if ch in ARABIC_LETTERS:
            diac = ""

            j= i+1
            while j < len(text) and text[j] in HARAKAT:
                diac += text[j]
                j+= 1

            pairs.append((ch,diac))
            i=j
        else:
            i +=1
    return  pairs



def compute_der(predicted:str,reference:str)->float:
    pred_pairs= extract_char_diacritic_pairs(predicted)
    ref_pairs= extract_char_diacritic_pairs(reference)

    n= min(len(pred_pairs), len(ref_pairs))

    if n== 0:
        return 1.0
    errors= sum(
        1 for (pc,pd ), (rc,rd) in zip(pred_pairs[:n], ref_pairs[:n])
        if pd !=rd # diacritic mismetch 
    )
    
    return errors / n 

def evaluate_corpus_der(predictions:List[str],
                        reference:List[str])-> Dict[str,float]:

                        ders= [compute_der(p,r) for p,r in zip(predictions,reference)]

                        return {"DER_mean": sum(ders) / len(ders),
                                "DER_min": min(ders),
                                "DER_max": max(ders),
                                "n_samples": len(ders),
                        }

