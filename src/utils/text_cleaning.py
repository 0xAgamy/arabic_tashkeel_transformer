import unicodedata
import re
ARABIC_LETTERS = set(chr(c) for c in range(0x0621, 0x064B))
HARAKAT = {
    "\u064E",   # fatḥa    
    "\u064F",   # ḍamma    
    "\u0650",   # kasra    
    "\u0651",   # shadda   
    "\u0652",   # sukun    
    "\u064B",   # fatḥatain 
    "\u064C",   # ḍammatain 
    "\u064D",   # kasratain 
    "\u0670",   # superscript alef 
}


def clean_arabic_text(text:str) -> str:
    """A Helper function to remove EPUB HTML tags, non-Arabic contenct and Harakat from the text  
    """
    text= unicodedata.normalize("NFKC",text)

    allowed= ARABIC_LETTERS | HARAKAT | {" ", "،", ".", "؟", "!", "،", "\n"}
    text= "".join(c for c in text if c in allowed or c.isspace())

    text= re.sub(r"\s+"," ",text).strip()
    return text


def strip_harakat(text:str)->str:
    """strip/remove harakat from text
    """
    return "".join(c for c in text if c not in HARAKAT)

