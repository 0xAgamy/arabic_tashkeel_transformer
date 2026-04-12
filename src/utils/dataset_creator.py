# here i will create the dataset, i will save the processed data to a json files 
import os ,json
from epub_loader import load_epub_pair,save_lists_to_json
from pathlib import Path

data_path= Path("data/raw")
output_path="data/processed/data.json" 
books_path_list=[str(p) for p in data_path.glob("*.epub")]
source, target=[] , []
for book_path in books_path_list:
    src,tgt=load_epub_pair(book_path)
    source.append(src)
    target.append(tgt)


save_lists_to_json(source,target,output_file=output_path)

