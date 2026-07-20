# Named Entity Recognition System

A comparative study and implementation of multiple deep learning architectures for Named Entity Recognition (NER), progressing from simple recurrent models to a fine-tuned transformer.

## Overview

This project implements and benchmarks four different approaches to Named Entity Recognition, evaluating how model architecture impacts sequence labeling performance:

1. **LSTM** — a baseline unidirectional recurrent model
2. **BiLSTM** — bidirectional LSTM to capture both past and future context
3. **BiLSTM + CRF** — bidirectional LSTM with a Conditional Random Field layer to model label dependencies
4. **Fine-tuned Transformer** — a pretrained transformer model fine-tuned for token classification

The fine-tuned transformer achieved the best overall performance, outperforming all recurrent-based architectures.

**Key finding:** The fine-tuned transformer outperformed all other architectures, benefiting from pretrained contextual representations that the LSTM-based models could not match even with bidirectionality and CRF-based label modeling.

## Project Structure

```
├── NER System.ipynb                   
├── best_model/
│   ├── config.json
│   ├── label_mapping.json
│   ├── model.savetensors
│   ├── tokenizer_config.json
│   └── tokenizer.json
└── README.md
```

## How to Run

```bash
python main.py

```

## Future Work

- Experiment with additional pretrained transformer backbones
- Domain adaptation to specialized text (e.g., medical, legal)
- Error analysis on misclassified entity spans
- Deployment as an API for real-time NER inference
