import os
import itertools
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer
)
import torch
import wandb
import numpy as np
from datetime import datetime
import json

from BARTScore.WMT.bart_score import BARTScorer

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")


import os
from datasets import load_dataset, load_from_disk, DatasetDict
tokenizer = AutoTokenizer.from_pretrained("t5-base")
DATA_DIR = "./data/alpaca_tokenized"

#
# 1) Load & format the RAW Alpaca once
#
raw = load_dataset("tatsu-lab/alpaca", split="train")

def format_with_context(example):
    inst = example["instruction"].strip()
    inp  = example["input"].strip()
    out  = example["output"].strip()
    src  = f"Context: {inp}\nResponse: {out}" if inp else out
    return {"input_text": src, "target_text": inst}

formatted = raw.map(format_with_context,
                    remove_columns=["instruction","input","output"])

#
# 2) Split the _formatted_ into train/test
#
split_formatted = formatted.train_test_split(test_size=0.1, seed=42)
train_formatted = split_formatted["train"]   # has input_text & target_text
eval_formatted  = split_formatted["test"]    # has input_text & target_text

#
# 3) Tokenize _those same_ splits (and drop the text columns)
#
def preprocess_fn(examples):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target_text"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tokenized = train_formatted.map(
    preprocess_fn, batched=True,
    remove_columns=["input_text","target_text"]
)
eval_tokenized  = eval_formatted.map(
    preprocess_fn, batched=True,
    remove_columns=["input_text","target_text"]
)

tokenized_dataset = DatasetDict({
    "train": train_tokenized,
    "test" : eval_tokenized
})

# cache to disk so you don’t pay the tokenization cost again
if not os.path.isdir(DATA_DIR):
    print(f"💾 Saving tokenized dataset to {DATA_DIR}")
    tokenized_dataset.save_to_disk(DATA_DIR)
else:
    print("📂 Loading tokenized dataset from disk…")
    tokenized_dataset = load_from_disk(DATA_DIR)

#
# 4) Prepare for Trainer and for zero/few‐shot separately
#
train_dataset      = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]

# this one is only for Trainer.predict:
eval_dataset_clean = eval_dataset.remove_columns(
    [c for c in eval_dataset.column_names
     if c not in ["input_ids","attention_mask","labels"]]
)

# this raw/textual one is for your zero/few‐shot routine:
eval_dataset_raw = eval_formatted

print(f"✔️ TRAIN size: {len(train_dataset)}")
print(f"✔️ EVAL (tokenized) size: {len(eval_dataset)}")
print(f"✔️ EVAL (raw/text) size: {len(eval_dataset_raw)}")





print(f"✔️ Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")


# Format and preprocess dataset
def format_with_context(example):
    instruction = example["instruction"].strip()
    input_field = example["input"].strip()
    output_field = example["output"].strip()
    input_text = f"Context: {input_field}\nResponse: {output_field}" if input_field else output_field
    return {"input_text": input_text, "target_text": instruction}

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=512,
        truncation=True,
        padding=False
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target_text"],
            max_length=128,
            truncation=True,
            padding=False
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def safe_decode_predictions(predictions, tokenizer, skip_special_tokens=True):
    """Safely decode model predictions: handles logits, token IDs, and invalid formats."""
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.array(predictions)

    if predictions.ndim == 3:
        print("⚠️ Detected logits. Applying argmax to get token IDs.")
        predictions = np.argmax(predictions, axis=-1)

    if not np.issubdtype(predictions.dtype, np.integer):
        raise ValueError("❌ Predictions must be integer token IDs. Got float or corrupt values.")

    predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)
    return tokenizer.batch_decode(predictions.tolist(), skip_special_tokens=skip_special_tokens)



def compute_metrics(preds, labels):
    import evaluate
    # Load metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # BARTScore
    #TODO - git clone https://github.com/neulab/BARTScore.git
    bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
    bart_scores = bart_scorer.score(preds, labels, batch_size=8)
    avg_bart = sum(bart_scores) / len(bart_scores)

    return {
        "bleu": bleu.compute(predictions=preds, references=[[label] for label in labels])["bleu"],
        "rouge1": rouge.compute(predictions=preds, references=labels)["rouge1"],
        "rougeL": rouge.compute(predictions=preds, references=labels)["rougeL"],
        "meteor": meteor.compute(predictions=preds, references=labels)["meteor"],
        "bart_score": avg_bart
    }

# Refrence-Free evaluation Part
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

def compute_perplexity(texts, model_id="gpt2"):
    """
    Compute average perplexity using a pre-trained GPT-2 model.

    Perplexity evaluates the fluency of generated text: lower values indicate more fluent output.
    This is a reference-free metric and does not require target ground truth.

    Args:
        texts (List[str]): List of generated texts to evaluate.
        model_id (str): HuggingFace model name or path to use for computing perplexity.

    Returns:
        List[float]: Perplexity scores for each text sample.
    """
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    model.eval()

    perplexities = []
    for text in texts:
        encodings = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)
    return perplexities


#TODO - pip install mauve-text
import mauve

def compute_mauve(p_texts, q_texts, device_id=0):
    """
    Compute MAUVE score between model predictions and input prompts.

    MAUVE estimates distributional similarity between two text sets (P: predictions, Q: inputs).
    It reflects how close the generated outputs are to human-like inputs without using references.

    Args:
        p_texts (List[str]): Generated texts.
        q_texts (List[str]): Input texts (or real samples).
        device_id (int): GPU device index (0 by default).

    Returns:
        float: MAUVE score (0–1), where higher is better.
    """
    # p_text: model outputs (predictions), q_text: inputs
    result = mauve.compute_mauve(p_text=p_texts, q_text=q_texts, device_id=device_id)
    return result.mauve


#TODO - pip install nltk
from nltk.translate.bleu_score import sentence_bleu

def compute_self_bleu(generations):
    """
    Compute Self-BLEU to estimate output diversity.

    This reference-free metric measures how similar each generated sample is to others.
    Lower scores indicate more diversity (less repetitive generations).

    Args:
        generations (List[str]): Generated texts.

    Returns:
        float: Average Self-BLEU score.
    """
    scores = []
    for i, candidate in enumerate(generations):
        references = generations[:i] + generations[i+1:]
        ref_tokens = [ref.split() for ref in references]
        cand_tokens = candidate.split()
        scores.append(sentence_bleu(ref_tokens, cand_tokens))
    return sum(scores) / len(scores)


def run_reference_free_metrics(predictions, inputs):
    """
    Run reference-free evaluation metrics: Perplexity, MAUVE, Self-BLEU.

    This function computes non-reference metrics that assess fluency, diversity, and distributional similarity.
    Useful for reporting quality of generation without comparing to ground-truth targets.

    Args:
        predictions (List[str]): Generated texts from the model.
        inputs (List[str]): Corresponding input prompts.

    Returns:
        Dict[str, float]: Dictionary of metric names and scores.
    """
    print("Running reference-free evaluation...")

    ppl = compute_perplexity(predictions)
    avg_ppl = sum(ppl) / len(ppl)

    mauve_score = compute_mauve(predictions, inputs)

    self_bleu = compute_self_bleu(predictions)

    return {
        "Perplexity (avg)": avg_ppl,
        "MAUVE": mauve_score,
        "Self-BLEU": self_bleu
    }


def run_zero_or_few_shot(model_name, eval_dataset, tokenizer, shots=0):
    """
    Runs zero-shot or few-shot inference on an evaluation dataset using a T5-like model.

    In zero-shot mode (shots=0), the model receives only the input prompt.
    In few-shot mode (shots>0), a small number of input-output examples from the dataset
    are prepended to each input to simulate few-shot prompting.

    Args:
        model_name (str): HuggingFace model name or path (e.g., "google/flan-t5-base").
        eval_dataset (Dataset): Evaluation dataset with "input_text" and "target_text" fields.
        tokenizer (AutoTokenizer): A tokenizer compatible with the model.
        shots (int): Number of few-shot examples to prepend (default is 0 = zero-shot).

    Returns:
        Tuple[List[str], List[str]]:
            - predictions (List[str]): Generated outputs for each example.
            - references (List[str]): Ground truth outputs from the dataset.

    Example:
        preds, refs = run_zero_or_few_shot("google/flan-t5-base", eval_dataset, tokenizer, shots=3)
    """
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    examples = []
    if shots > 0:
        for i in range(shots):
            input_text = eval_dataset[i]["input_text"]
            target_text = eval_dataset[i]["target_text"]
            examples.append(f"{target_text}\nInstruction: {input_text}")

    prompts = []
    for example in eval_dataset:
        shot_text = "\n\n".join(examples) if shots > 0 else ""
        full_input = f"{shot_text}\n\n{example['target_text']}\nInstruction:"
        prompts.append(full_input.strip())

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64)

    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    references = [ex["target_text"] for ex in eval_dataset]

    return predictions, references
import torch
torch.cuda.empty_cache()

if __name__ == "__main__":

    # with open("./results/best_hyperparameters.json", "r") as f:
    #     best_config = json.load(f)

    model_checkpoint = "/cs/labs/werman/dbusbib123/projects/results/best_run/checkpoint-4389"
    print(f"Loading best model from: {model_checkpoint}")

    # Check what files exist in the root directory
    print(f"📁 Files in root directory: {os.listdir(model_checkpoint)}")

    # Look for model files in root directory (where trainer.save_model() saves them)
    model_files = ["pytorch_model.bin", "model.safetensors"]
    has_model_in_root = any(os.path.exists(os.path.join(model_checkpoint, file)) for file in model_files)

    if has_model_in_root:
        print(f"✅ Found model in root directory: {model_checkpoint}")
    else:
        print(f"❌ No model files found. Falling back to base model...")
        # model_checkpoint = "t5-base"

    print(f"📦 Actually loading model from: {model_checkpoint}")
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    print(model.config)  
# e.g. shows model_type, d_model, num_layers, vocab_size, etc.

    # Prepare trainer for evaluation
    eval_args = Seq2SeqTrainingArguments(
        output_dir="./results/eval",
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        gradient_checkpointing=True,
        do_predict=True,
        report_to="none"
        ,fp16=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    # model.gradient_checkpointing_enable()

    print("🔍 Running final evaluation...")
    outputs = trainer.predict(eval_dataset_clean)

    decoded_preds = safe_decode_predictions(outputs.predictions, tokenizer)
    decoded_labels = safe_decode_predictions(outputs.label_ids, tokenizer)

    print("📊 Computing reference-based metrics...")
    # metrics = compute_metrics(decoded_preds, decoded_labels)
    # print(json.dumps(metrics, indent=2))

    # print("📊 Computing reference-free metrics...")
    # decoded_inputs = safe_decode_predictions([ex["input_ids"] for ex in eval_dataset], tokenizer)
    # ref_free_metrics = run_reference_free_metrics(decoded_preds, decoded_inputs)
    # print(json.dumps(ref_free_metrics, indent=2))

    # # Merge metrics into one file
    # all_metrics = {**metrics, **ref_free_metrics}

    print("\n🚀 Running Zero-shot Evaluation...")
    #todo - maybe add flan-t5-base or similar for zero-shot
    small_eval = eval_dataset_raw.select(range(1000))
    zero_preds, zero_refs = run_zero_or_few_shot("t5-base", small_eval, tokenizer, shots=0)
    # torch.cuda.empty_cache()
    zero_metrics = compute_metrics(zero_preds, zero_refs)
    zero_ref_free = run_reference_free_metrics(zero_preds, [ex["input_text"] for ex in eval_dataset])
    
    
    with open("./results/eval/zero_shot_metrics.json", "w") as f:
        json.dump({**zero_metrics, **zero_ref_free}, f, indent=2)

    print("\n🚀 Running Few-shot Evaluation (e.g., 3-shot)...")
    #todo - maybe add flan-t5-base or similar for few-shot
    few_preds, few_refs = run_zero_or_few_shot("t5-base", small_eval, tokenizer, shots=3)
    few_metrics = compute_metrics(few_preds, few_refs)
    few_ref_free = run_reference_free_metrics(few_preds, [ex["input_text"] for ex in eval_dataset])
    with open("./results/eval/few_shot_metrics.json", "w") as f:
        json.dump({**few_metrics, **few_ref_free}, f, indent=2)

    results_summary = {
        # "finetuned": all_metrics,
        "zero_shot": {**zero_metrics, **zero_ref_free},
        "few_shot": {**few_metrics, **few_ref_free}
    }
    with open("./results/eval/summary_metrics.json", "w") as f:
        json.dump(results_summary, f, indent=2)
        
        
        
