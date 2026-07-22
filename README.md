# 🔎 Named Entity Recognition System

A deep learning system that identifies and classifies named entities in text (such as persons, organizations, and locations). The project includes a complete training pipeline comparing multiple sequence labeling architectures, from recurrent networks to a fine-tuned transformer.

## Project Goal

The goal of this project is to compare several deep learning architectures for Named Entity Recognition and identify the best-performing approach.

Four models were trained and evaluated:

| Model | Type | Notes |
|---|---|---|
| LSTM | Trained from scratch | Unidirectional recurrent baseline, processes tokens using only past context |
| BiLSTM | Trained from scratch | Bidirectional recurrent model, captures both past and future context |
| BiLSTM + CRF | Trained from scratch | BiLSTM with a CRF layer on top to model dependencies between adjacent tags |
| Fine-tuned Transformer | Transfer learning | Pretrained transformer fine-tuned for token classification — best performing model |

## Highlights

- **Data preparation:** The dataset was split into training, validation, and test sets before preprocessing, with tokens tagged in BIO format.
- **Model comparison:** Models were evaluated on precision, recall, and F1-score, both overall and per entity type.
- **Progressive architecture study:** Each model builds on the limitations of the previous one — from a unidirectional LSTM, to bidirectional context (BiLSTM), to structured label prediction (BiLSTM + CRF), to contextual pretrained representations (fine-tuned transformer).
- **Best model:** The fine-tuned transformer achieved the strongest results across all metrics and entity types.

## Project Structure

```
├── NER System.ipynb                       
├── main.py
└── README.md
```

## Built With

- Tensorflow/Keras — model building & training
- Hugging Face Transformers — pretrained transformer backbone & fine-tuning
- torchcrf / pytorch-crf — CRF layer for the BiLSTM + CRF model
- NumPy, Pandas, Matplotlib, Seaborn — data processing & visualization

## Quick Start

```bash
pip install tensorflow transformers scikit-learn pandas numpy
python main.py
```

## Results

The fine-tuned transformer outperformed the LSTM, BiLSTM, and BiLSTM + CRF models, benefiting from pretrained contextual representations that the recurrent-based architectures could not match even with bidirectionality and CRF-based label modeling.
