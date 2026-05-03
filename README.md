# Arabic Tashkeel Transformer
PyTorch implementation of a character-level Seq2Seq Transformer for Arabic Tashkeel (diacritization).

##  Features

- **Character-Level Tokenization** : Exact alignment between input/output; no subword fragmentation of diacritics


- **Transformer Encoder-Decoder**: From-scratch implementation with multi-head attention & positional encoding
	
- **Beam Search Decoding**: Explore multiple hypotheses for higher-quality diacritization
- **Greedy Search**: Fast, lightweight inference option alongside beam 

- **Smart Text Chunking**: Handle arbitrarily long inputs with sentence-aware splitting

- **FastAPI REST API** :Production-ready endpoint with async support & error handling

- **DER Evaluation**:Diacritic Error Rate metric aligned on base characters


## Installation
### Requirements
```bash
pip install -r requirements.txt
```
## Project Structre
```bash
├── app
│   ├── main.py
│   ├── models.py
├── assets
│   ├── config.json
│   └── tokenizer.json
├── checkpoints
│   └── best_model.pt
├── data
│   ├── processed
│   │   └── data.json
├── images
│   └── arc.png
├── LICENSE
├── notebooks
│   └── arabic-tashkeel.ipynb
├── postman_collection
│   └── Arabic-Tashkeel.postman_collection.json
├── README.md
├── requirements.txt
├── scripts
│   └── train.py
├── src
│   ├── data
│   │   ├── dataset.py
│   │   └── tokenizer.py
│   ├── evaluation
│   │   └── metrics.py
│   ├── inference
│   │   ├── beam_search.py
│   │   ├── greedy_decode.py
│   │   ├── predictor.py
│   ├── model
│   │   ├── positional_encoding.py
│   │   └── transformer.py
│   ├── training
│   │   ├── scheduler.py
│   │   └── trainer.py
│   └── utils
│       ├── dataset_creator.py
│       ├── epub_loader.py
│       ├── text_chunker.py
│       └── text_cleaning.py
└── tests
    └── test_tokenizer.py
```

## Model Architecture
```bash
┌─────────────────────────────────────┐
│  INPUT: "كتب الطالب الدرس"          │
└────────────────┬────────────────────┘
                 ▼
┌─────────────────────────────────────┐
│  CHARACTER TOKENIZER                │
│  [ك, ت, ب, _,ا, ل, ط, ...] → IDs    │
└────────────────┬────────────────────┘
                 ▼
┌─────────────────────────────────────┐
│  ENCODER (3 Layers)                 │
│  • Embedding (d_model=256)          │
│  • Positional Encoding (sinusoidal) │
│  • Multi-Head Self-Attention (8h)   │
│  • Feed-Forward Network             │
└────────────────┬────────────────────┘
                 ▼
┌─────────────────────────────────────┐
│  DECODER (3 Layers)                 │
│  • Masked Self-Attention (causal)   │
│  • Cross-Attention ← Encoder Memory │
│  • Feed-Forward Network             │
└────────────────┬────────────────────┘
                 ▼
┌─────────────────────────────────────┐
│  OUTPUT HEAD                        │
│  Linear(d_model → vocab_size)       │
│  Softmax → Next-token distribution  │
└────────────────┬────────────────────┘
                 ▼
┌─────────────────────────────────────┐
│  DECODING                           │
│  • Greedy: argmax at each step      │
│  • Beam Search (k=4): top-k paths   │
└────────────────┬────────────────────┘
                 ▼
┌─────────────────────────────────────┐
│  OUTPUT: "كَتَبَ الطَّالِبُ الدَّرْسَ" │
└─────────────────────────────────────┘

```
## Dataset
- we collect dataset from `أولو العلم `Telegram channel for providing high-quality diacritized Arabic EPUBs

## Evaluation

**Diacritic Error Rate (DER)**
The standard metric for tashkeel tasks:

- Lower is better (0.0 = perfect)
- Computed by aligning prediction/reference on base characters
- Ignores insertion/deletion of base characters (focuses on diacritic accuracy)

### Results
- **Prototype (11 EPUBs)** : 8.6% DER
- **Expanded (70 EPUBs)** : 3.5% DER , & 2.0 % CER
## Postman collection

-  [Download post man collection](/postman_collection/Arabic-Tashkeel.postman_collection.json)

##  Acknowledgments

- أولو العلم Telegram channel for providing high-quality diacritized Arabic EPUBs


*References*

- Vaswani et al. (2017). Attention Is All You Need. arXiv:1706.03762
- Antoun et al. (2020). AraBERT: Transformer-based Model for Arabic Language Understanding. arXiv:2003.00104
- Inoue et al. (2021). Diacritization of Arabic Text Using Deep Learning. ACL 2021
---

 Made with ❤️ for the Arabic NLP community
