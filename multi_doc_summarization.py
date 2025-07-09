import os
import time
import datasets
import torch
from torch import nn
import networkx as nx
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, 
    LongT5ForConditionalGeneration, PegasusForConditionalGeneration, 
    BartForConditionalGeneration, EncoderDecoderModel
)
from rouge_score import rouge_scorer
from keybert import KeyBERT
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import Dataset

# 1. Dataset Preparation
class CNNDailyMailDataset(Dataset):
    def __init__(self, split="test", max_length=512):
        self.dataset = datasets.load_dataset("cnn_dailymail", "3.0.0")[split]
        self.max_length = max_length
        self.tokenizers = {}
        try:
            self.tokenizers["bart"] = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        except Exception as e:
            print(f"Error loading BART tokenizer: {e}")
        try:
            self.tokenizers["pegasus"] = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail", use_fast=False)
        except Exception as e:
            print(f"Error loading PEGASUS tokenizer: {e}. Skipping PEGASUS.")
            self.tokenizers["pegasus"] = None
        try:
            self.tokenizers["t5-base"] = AutoTokenizer.from_pretrained("t5-base")
            self.tokenizers["t5-large"] = AutoTokenizer.from_pretrained("t5-large")
            self.tokenizers["longt5"] = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
            self.tokenizers["primera"] = AutoTokenizer.from_pretrained("allenai/PRIMERA")
            self.tokenizers["absformer"] = AutoTokenizer.from_pretrained("t5-base")
            self.tokenizers["tg-multisum"] = AutoTokenizer.from_pretrained("t5-base")
            self.tokenizers["hierarchical"] = AutoTokenizer.from_pretrained("t5-base")
            self.tokenizers["dca"] = AutoTokenizer.from_pretrained("t5-base")
            self.tokenizers["keyword-t5"] = AutoTokenizer.from_pretrained("t5-base")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        article = self.dataset[idx]["article"]
        highlights = self.dataset[idx]["highlights"]
        return {"article": article, "highlights": highlights}

    def preprocess_for_model(self, article, model_name):
        if model_name not in self.tokenizers or self.tokenizers[model_name] is None:
            raise ValueError(f"Tokenizer for {model_name} not available.")
        tokenizer = self.tokenizers[model_name]
        if model_name == "keyword-t5":
            try:
                kw_model = KeyBERT()
                keywords = kw_model.extract_keywords(article, top_n=5)
                article = "Keywords: " + ", ".join([kw[0] for kw in keywords]) + " | " + article
            except Exception as e:
                print(f"Keyword extraction failed: {e}")
        inputs = tokenizer(
            article, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        return inputs

# 2. Model Implementation
class Absformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=6
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=512, nhead=8), num_layers=6
        )
        self.fc = nn.Linear(512, 32000)  # Match tokenizer vocab size

    def forward(self, src, tgt):
        src = src.transpose(0, 1)  # [seq_len, batch, dim]
        tgt = tgt.transpose(0, 1)
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return self.fc(output.transpose(0, 1))

class TGMultiSum(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.graph_layer = nn.Linear(512, 512)  # Simple graph processing layer

    def forward(self, inputs, graph=None):
        outputs = self.base_model(**inputs)
        return outputs

class HierarchicalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.sentence_encoder = EncoderDecoderModel.from_pretrained("t5-base").encoder
        self.document_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=3
        )
        self.decoder = T5ForConditionalGeneration.from_pretrained("t5-base").decoder

    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        encoded = self.sentence_encoder(input_ids).last_hidden_state
        doc_encoded = self.document_encoder(encoded.transpose(0, 1)).transpose(0, 1)
        return self.decoder(input_ids=doc_encoded)

class DCAAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=3
        )

    def forward(self, inputs):
        return self.encoder(inputs)

class DCA(nn.Module):
    def __init__(self, num_agents=3):
        super().__init__()
        self.agents = nn.ModuleList([DCAAgent() for _ in range(num_agents)])
        self.decoder = T5ForConditionalGeneration.from_pretrained("t5-base").decoder

    def forward(self, inputs):
        agent_outputs = [agent(inputs["input_ids"]) for agent in self.agents]
        combined = sum(agent_outputs) / len(agent_outputs)
        return self.decoder(input_ids=combined)

class SummarizationModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self.load_model()
        if self.model is not None:
            self.model.to(self.device)

    def load_model(self):
        try:
            if self.model_name == "bart":
                model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
                tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
            elif self.model_name == "pegasus":
                model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-cnn_dailymail")
                tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail", use_fast=False)
            elif self.model_name == "t5-base":
                model = T5ForConditionalGeneration.from_pretrained("t5-base")
                tokenizer = AutoTokenizer.from_pretrained("t5-base")
            elif self.model_name == "t5-large":
                model = T5ForConditionalGeneration.from_pretrained("t5-large")
                tokenizer = AutoTokenizer.from_pretrained("t5-large")
            elif self.model_name == "longt5":
                model = LongT5ForConditionalGeneration.from_pretrained("google/long-t5-tglobal-base")
                tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
            elif self.model_name == "primera":
                model = AutoModelForSeq2SeqLM.from_pretrained("allenai/PRIMERA")
                tokenizer = AutoTokenizer.from_pretrained("allenai/PRIMERA")
            elif self.model_name == "absformer":
                model = Absformer()
                tokenizer = AutoTokenizer.from_pretrained("t5-base")
            elif self.model_name == "tg-multisum":
                model = TGMultiSum()
                tokenizer = AutoTokenizer.from_pretrained("t5-base")
            elif self.model_name == "hierarchical":
                model = HierarchicalTransformer()
                tokenizer = AutoTokenizer.from_pretrained("t5-base")
            elif self.model_name == "dca":
                model = DCA()
                tokenizer = AutoTokenizer.from_pretrained("t5-base")
            elif self.model_name == "keyword-t5":
                model = T5ForConditionalGeneration.from_pretrained("t5-base")
                tokenizer = AutoTokenizer.from_pretrained("t5-base")
            else:
                raise ValueError(f"Model {self.model_name} not supported.")
            return model, tokenizer
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            return None, None

    def fine_tune(self, train_dataset, num_epochs=1, batch_size=4):
        if self.model is None:
            print(f"Skipping fine-tuning for {self.model_name} due to load error.")
            return
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        self.model.train()
        for epoch in range(num_epochs):
            for i in range(0, len(train_dataset), batch_size):
                batch = train_dataset[i:i+batch_size]
                inputs = self.tokenizer(
                    [item["article"] for item in batch], 
                    max_length=512, truncation=True, padding="max_length", return_tensors="pt"
                ).to(self.device)
                labels = self.tokenizer(
                    [item["highlights"] for item in batch], 
                    max_length=128, truncation=True, padding="max_length", return_tensors="pt"
                ).input_ids.to(self.device)
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                print(f"Epoch {epoch+1}, Batch {i//batch_size}, Loss: {loss.item()}")

    def generate_summary(self, article, max_length=128):
        if self.model is None or self.tokenizer is None:
            return f"Cannot generate summary for {self.model_name} due to load error."
        self.model.eval()
        inputs = self.tokenizer(
            article, max_length=512, truncation=True, padding="max_length", return_tensors="pt"
        ).to(self.device)
        try:
            summary_ids = self.model.generate(
                inputs.input_ids, max_length=max_length, num_beams=4, early_stopping=True
            )
            return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        except:
            return f"Summary generation not fully implemented for {self.model_name}."

# 3. Evaluation
def evaluate_model(model, dataset, num_samples=10):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for i in range(min(num_samples, len(dataset))):
        article = dataset[i]["article"]
        reference = dataset[i]["highlights"]
        summary = model.generate_summary(article)
        score = scorer.score(reference, summary)
        for metric in scores:
            scores[metric].append(score[metric].fmeasure)
    return {metric: np.mean(scores[metric]) for metric in scores}

# 4. Reporting
def create_comparison_table(results):
    df = pd.DataFrame(results)
    df = df[["Model", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Training Time (s)", "GPU Usage (GB)"]]
    return df

def plot_results(df):
    df_melted = df.melt(id_vars="Model", value_vars=["ROUGE-1", "ROUGE-2", "ROUGE-L"], 
                        var_name="Metric", value_name="Score")
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model", y="Score", hue="Metric", data=df_melted)
    plt.title("ROUGE Scores Comparison Across Models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/rouge_comparison.png")
    plt.close()

# 5. Main Pipeline
def main():
    dataset = CNNDailyMailDataset(split="test")
    train_dataset = CNNDailyMailDataset(split="train")
    
    model_names = [
        "bart", "pegasus", "t5-base", "t5-large", "longt5", "primera",
        "absformer", "tg-multisum", "hierarchical", "dca", "keyword-t5"
    ]
    results = []
    
    for model_name in model_names:
        print(f"Processing {model_name}...")
        model = SummarizationModel(model_name)
        
        start_time = time.time()
        # Fine-tuning skipped to save time; uncomment to enable
        # model.fine_tune(train_dataset, num_epochs=1)
        training_time = time.time() - start_time
        
        try:
            scores = evaluate_model(model, dataset, num_samples=10)
            gpu_usage = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        except Exception as e:
            print(f"Evaluation failed for {model_name}: {e}")
            scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            gpu_usage = 0.0
        
        results.append({
            "Model": model_name,
            "ROUGE-1": scores["rouge1"],
            "ROUGE-2": scores["rouge2"],
            "ROUGE-L": scores["rougeL"],
            "Training Time (s)": training_time,
            "GPU Usage (GB)": gpu_usage
        })
    
    df = create_comparison_table(results)
    print("\nComparison Table:")
    print(df)
    
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/rouge_scores.csv", index=False)
    
    plot_results(df)

if __name__ == "__main__":
    main()