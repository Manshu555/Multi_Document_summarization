# Multi-Document Summarization Project

This project evaluates multiple text summarization models on the CNN/Daily Mail dataset, comparing their performance using ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L). The models include pre-trained models (BART, PEGASUS, T5-base, T5-large, LongT5, PRIMERA, Keyword-T5) and custom implementations (Absformer, TG-MultiSum, Hierarchical Transformer, DCA). Results are saved as a CSV file (`results/rouge_scores.csv`) and visualized in a bar plot (`results/rouge_comparison.png`).

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Known Issues](#known-issues)
- [Future Improvements](#future-improvements)
- [License](#license)

## Project Overview
The script `multi_doc_summarization.py` loads the CNN/Daily Mail dataset (version 3.0.0) and evaluates 11 summarization models on 10 test samples. It computes ROUGE scores, training time, and GPU usage, saving results to `results/rouge_scores.csv` and generating a bar plot in `results/rouge_comparison.png`. The dataset and pre-trained models are automatically downloaded via the `datasets` and `transformers` libraries, requiring no manual downloads.

### Models Evaluated
- **Pre-trained Models**:
  - BART (`facebook/bart-large-cnn`)
  - PEGASUS (`google/pegasus-cnn_dailymail`)
  - T5-base (`t5-base`)
  - T5-large (`t5-large`)
  - LongT5 (`google/long-t5-tglobal-base`)
  - PRIMERA (`allenai/PRIMERA`)
  - Keyword-T5 (T5-base with keyword augmentation using KeyBERT)
- **Custom Models** (placeholders, not fully implemented):
  - Absformer
  - TG-MultiSum
  - Hierarchical Transformer
  - DCA (Divide-and-Conquer Agents)

## Requirements
- Python 3.12
- Dependencies listed in `requirements.txt`:
  ```
  torch>=1.9.0
  transformers>=4.20.0
  datasets>=2.0.0
  rouge-score>=0.1.2
  pandas>=1.4.0
  matplotlib>=3.5.0
  seaborn>=0.11.0
  numpy>=1.21.0
  networkx>=2.6.0
  keybert>=0.5.0
  tiktoken>=0.5.0
  protobuf>=3.20.0
  sentencepiece>=0.1.99
  ```

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Manshu555/Multi_Document_summarization.git
   cd Multi_Document_summarization
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv summarization_env
   source summarization_env/bin/activate  # macOS/Linux
   # On Windows: summarization_env\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Key Dependencies**:
   ```bash
   pip show sentencepiece tiktoken protobuf
   ```

5. **Ensure Disk Space**:
   - ~15GB for models (~10–12GB) and dataset (~1GB), cached in `~/.cache/huggingface/`.

## Usage
1. **Run the Script**:
   ```bash
   python multi_doc_summarization.py
   ```
   - The script:
     - Downloads the CNN/Daily Mail dataset and pre-trained models (if not cached).
     - Evaluates 11 models on 10 test samples.
     - Saves ROUGE scores to `results/rouge_scores.csv`.
     - Generates a bar plot in `results/rouge_comparison.png`.

2. **Enable Fine-Tuning (Optional)**:
   - Uncomment `model.fine_tune(train_dataset, num_epochs=1)` in `main()` to fine-tune models.
   - Requires significant compute (GPU recommended).

3. **Reduce Resource Usage**:
   - Skip large models (e.g., T5-large) by editing `model_names` in `main()`:
     ```python
     model_names = ["bart", "pegasus", "t5-base", "longt5", "primera", "absformer", "tg-multisum", "hierarchical", "dca", "keyword-t5"]
     ```
   - Reduce samples in `evaluate_model`:
     ```python
     def evaluate_model(model, dataset, num_samples=5):
     ```

## Project Structure
```
Multi_Document_summarization/
├── multi_doc_summarization.py      # Main script
├── requirements.txt                # Dependencies
├── README.md                       # Project documentation
├── docs/
│   └── pipeline_documentation.md    # Detailed pipeline notes
├── results/
│   ├── rouge_scores.csv            # ROUGE scores, training time, GPU usage
│   └── rouge_comparison.png        # Bar plot of ROUGE scores
└── .gitignore                      # Excludes virtual env and cache
```

## Results
- **ROUGE Scores**: Saved in `results/rouge_scores.csv`. Example (approximate):
  - BART: ROUGE-1 ~0.40, ROUGE-2 ~0.18, ROUGE-L ~0.35
  - PEGASUS: ROUGE-1 ~0.35, ROUGE-2 ~0.15, ROUGE-L ~0.30
  - Custom models (Absformer, TG-MultiSum, Hierarchical, DCA): ~0.0 (placeholders)
- **Visualization**: `results/rouge_comparison.png` shows a bar plot comparing ROUGE-1, ROUGE-2, and ROUGE-L across models.
- **Performance**:
  - Pre-trained models (BART, PEGASUS, etc.) perform well without fine-tuning.
  - Custom models require full implementation for competitive scores.

## Known Issues
- **PEGASUS Warning**:
  - Uninitialized weights (`model.decoder.embed_positions.weight`, `model.encoder.embed_positions.weight`) when loading `google/pegasus-cnn_dailymail`. Functional for inference but may benefit from fine-tuning.
- **Custom Models**:
  - Absformer, TG-MultiSum, Hierarchical Transformer, and DCA are placeholders, yielding low ROUGE scores (~0.0).
- **Resource Usage**:
  - T5-large and LongT5 may cause memory issues on low-memory GPUs or CPUs. Skip or reduce samples as needed.
- **Tokenizer Issues**:
  - PEGASUS tokenizer required `use_fast=False` to avoid `tiktoken` errors. Ensure `sentencepiece` is installed.

## Future Improvements
- Implement full versions of custom models:
  - **Absformer**: Add unsupervised sentence masking.
  - **TG-MultiSum**: Build sentence graph with `networkx`.
  - **Hierarchical Transformer**: Encode sentences and documents separately.
  - **DCA**: Add agent communication logic.
- Fine-tune pre-trained models on the CNN/Daily Mail dataset.
- Evaluate on more samples (e.g., 100) for robust ROUGE scores.
- Optimize memory usage for large models (e.g., T5-large).
- Add support for other datasets (e.g., Multi-News).

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.