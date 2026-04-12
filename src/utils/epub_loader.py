import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from typing import Tuple ,List,  Optional
import re
import json
from text_cleaning import clean_arabic_text, strip_harakat






def extract_text_from_epub(epub_path):
    """Extract text from epub books, and clean it 
    Return:
        a list of clean text
    """
    book= epub.read_epub(epub_path)
    full_text=[]


    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup= BeautifulSoup(item.get_body_content(),"html.parser")
            text= soup.get_text(separator=" ")
            text= clean_arabic_text(text)
            if len(text) > 20 :
                full_text.append(text)
    return " ".join(full_text)


def load_epub_pair(path:str, max_chars: Optional[int]=None)-> Tuple[List[str], List[str]]:
    raw = extract_text_from_epub(path)
    if max_chars:
        raw = raw[:max_chars]
    
    sentences_diac = re.split(r"[.؟!،\n]+", raw)
    sentences_diac= [s.strip() for s in sentences_diac if len(s.strip()) > 5 ]

    source= [strip_harakat(s) for s in sentences_diac]
    target= sentences_diac
    return source,target



def save_lists_to_json(src_list, tgt_list, output_file):
    """
    Save two lists (src, tgt) as paired JSON records.

    Args:
        src_list (list[str]): source sentences
        tgt_list (list[str]): target sentences
        output_file (str): path to output JSON file
    """
    if len(src_list) != len(tgt_list):
        raise ValueError("src_list and tgt_list must have the same length")

    data = [
        {"src": src, "tgt": tgt}
        for src, tgt in zip(src_list, tgt_list)
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)    
    