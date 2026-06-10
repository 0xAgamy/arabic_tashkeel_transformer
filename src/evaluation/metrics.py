from typing import List, Tuple, Dict
import re 
import unicodedata
import editdistance


def extract_base_and_diacritics(text: str) -> List[Tuple[str, str]]:
    """
    Extract (base_char, diacritics) pairs from Arabic text.
    Handles shadda+haraka combinations correctly.
    
    Returns: List of tuples like 
    """
 
    BASE_PATTERN = re.compile(r'[\u0621-\u064A]')  
    DIACRITIC_PATTERN = re.compile(r'[\u064B-\u0652\u0670]')  
    
    pairs = []
    i = 0
    while i < len(text):
        char = text[i]
        if BASE_PATTERN.match(char):
            
            diacritics = ""
            j = i + 1
            while j < len(text) and DIACRITIC_PATTERN.match(text[j]):
                diacritics += text[j]
                j += 1
            pairs.append((char, diacritics))
            i = j
        else:
            
            pairs.append((char, ""))
            i += 1
    return pairs



def align_on_base_chars(
    pred_pairs: List[Tuple[str, str]],
    ref_pairs: List[Tuple[str, str]]
) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
    """
    Align prediction and reference pairs on BASE CHARACTERS using greedy alignment.
    
    Returns: List of (pred_pair, ref_pair) tuples, 
    """
    aligned = []
    i, j = 0, 0
    
    while i < len(pred_pairs) or j < len(ref_pairs):
        
        if i >= len(pred_pairs) and j >= len(ref_pairs):
            break
        
        
        if i >= len(pred_pairs):
            aligned.append((None, ref_pairs[j]))
            j += 1
            continue
        
        
        if j >= len(ref_pairs):
            aligned.append((pred_pairs[i], None))
            i += 1
            continue
        
        pred_base, pred_diac = pred_pairs[i]
        ref_base, ref_diac = ref_pairs[j]
        
        
        if pred_base == ref_base:
            aligned.append((pred_pairs[i], ref_pairs[j]))
            i += 1
            j += 1
        else:
            
            window = 3
            found = False
            
            
            for k in range(1, min(window + 1, len(pred_pairs) - i)):
                if pred_pairs[i + k][0] == ref_base:
                    
                    for skip in range(k):
                        aligned.append((pred_pairs[i + skip], None))
                    i += k
                    found = True
                    break
            
            if not found:
                
                for k in range(1, min(window + 1, len(ref_pairs) - j)):
                    if ref_pairs[j + k][0] == pred_base:
                        
                        for skip in range(k):
                            aligned.append((None, ref_pairs[j + skip]))
                        j += k
                        found = True
                        break
            
            if not found:
                
                aligned.append((pred_pairs[i], ref_pairs[j]))
                i += 1
                j += 1
    
    return aligned

def compute_der(predicted: str, reference: str) -> float:
    """
    Compute Diacritic Error Rate with BASE-CHARACTER ALIGNMENT.    
    """
    if not reference.strip():
        return 0.0 if not predicted.strip() else 1.0
    
    pred_pairs = extract_base_and_diacritics(predicted)
    ref_pairs = extract_base_and_diacritics(reference)
    
    ref_base_count = sum(1 for base, _ in ref_pairs if base.strip())
    if ref_base_count == 0:
        return 0.0
    
    aligned = align_on_base_chars(pred_pairs, ref_pairs)
    
    errors = 0
    for pred_pair, ref_pair in aligned:
        if pred_pair is None:
            errors += 1
        elif ref_pair is None:
            errors += 1
        else:
            pred_base, pred_diac = pred_pair
            ref_base, ref_diac = ref_pair
            if pred_base != ref_base:
                errors += 1  
            elif pred_diac != ref_diac:
                errors += 1  
    
    return errors / ref_base_count

def normalize_arabic(text: str) -> str:
    """Normalize Unicode, collapse spaces, and strip control chars for fair comparison."""
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())  # collapse multiple spaces
    return text

def exact_match(predictions: List[str], references: List[str]) -> float:
    """Proportion of exactly matching prediction-reference pairs."""
    if not predictions:
        return 0.0
    norm_preds = [normalize_arabic(p) for p in predictions]
    norm_refs = [normalize_arabic(r) for r in references]
    correct = sum(p == r for p, r in zip(norm_preds, norm_refs))
    return correct / len(predictions)

def cer(predicted: str, reference: str) -> float:
    """
    Character Error Rate (CER) using Levenshtein distance.
    CER = (S + D + I) / N_ref
    """
    ref = normalize_arabic(reference)
    pred = normalize_arabic(predicted)
    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return editdistance.eval(pred, ref) / len(ref)

def compute_corpus_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Unified evaluation: DER + CER + Exact Match."""
    
    
    der_scores = [compute_der(p, r) for p, r in zip(predictions, references)]
    cer_scores = [cer(p, r) for p, r in zip(predictions, references)]
    em = exact_match(predictions, references)
    return {
        "DER_mean": sum(der_scores) / len(der_scores),
        "DER_std":  (sum((d - sum(der_scores)/len(der_scores))**2 for d in der_scores) / len(der_scores))**0.5,
        "CER_mean": sum(cer_scores) / len(cer_scores),
        "Exact_Match": em,
        "n_samples": len(predictions)}

