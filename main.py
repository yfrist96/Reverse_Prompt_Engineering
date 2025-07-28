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

# CPU-specific optimizations
if device.type == "cpu":
    print("💡 CPU detected - applying CPU optimizations:")
    print("  - Reduced batch sizes")
    print("  - Disabled gradient checkpointing")
    print("  - Using smaller models where possible")
    
    # Set CPU threads for better performance
    torch.set_num_threads(4)  # Adjust based on your CPU cores


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
    results = {}
    
    # Clean predictions and labels
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    print(f"Computing metrics for {len(preds)} predictions...")

    # BLEU
    try:
        bleu = evaluate.load("bleu")
        results["bleu"] = bleu.compute(predictions=preds, references=[[label] for label in labels])["bleu"]
        print(f"✅ BLEU: {results['bleu']:.4f}")
    except Exception as e:
        print(f"❌ Error computing BLEU: {e}")
        results["bleu"] = None

    # ROUGE
    try:
        rouge = evaluate.load("rouge")
        rouge_scores = rouge.compute(predictions=preds, references=labels)
        results["rouge1"] = rouge_scores["rouge1"]
        results["rougeL"] = rouge_scores["rougeL"]
        print(f"✅ ROUGE-1: {results['rouge1']:.4f}, ROUGE-L: {results['rougeL']:.4f}")
    except Exception as e:
        print(f"❌ Error computing ROUGE: {e}")
        results["rouge1"] = None
        results["rougeL"] = None

    # METEOR
    try:
        meteor = evaluate.load("meteor")
        results["meteor"] = meteor.compute(predictions=preds, references=labels)["meteor"]
        print(f"✅ METEOR: {results['meteor']:.4f}")
    except Exception as e:
        print(f"❌ Error computing METEOR: {e}")
        results["meteor"] = None

    # BARTScore
    try:
        print("Computing BARTScore...")
        # Check for empty predictions first
        if not preds or not labels:
            print("❌ Empty predictions or labels for BARTScore")
            results["bart_score"] = None
        else:
            # Filter out empty strings
            valid_pairs = [(p, l) for p, l in zip(preds, labels) if p.strip() and l.strip()]
            if not valid_pairs:
                print("❌ No valid text pairs for BARTScore")
                results["bart_score"] = None
            else:
                valid_preds, valid_labels = zip(*valid_pairs)
                # Use smaller batch size for CPU
                batch_size = 1 if device.type == "cpu" else 2
                bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
                bart_scores = bart_scorer.score(list(valid_preds), list(valid_labels), batch_size=batch_size)
                avg_bart = sum(bart_scores) / len(bart_scores)
                results["bart_score"] = avg_bart
                print(f"✅ BARTScore: {avg_bart:.4f}")
                # Clean up BARTScore model
                del bart_scorer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Error computing BARTScore: {e}")
        results["bart_score"] = None

    return results

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
    # Load models fresh each time to avoid memory conflicts
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    model.eval()

    perplexities = []
    try:
        for i, text in enumerate(texts):
            if i % 20 == 0:  # Progress indicator
                print(f"  Processing text {i+1}/{len(texts)}")
            
            # Skip empty texts
            if not text.strip():
                perplexities.append(float('inf'))
                continue
                
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**encodings, labels=encodings["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)
                
            # Clear GPU memory periodically
            if i % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        # Clean up model from memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
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
import nltk
import ssl

# Handle SSL certificate issues on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        print(f"Failed to download punkt: {e}")

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"Failed to download punkt_tab: {e}")

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK wordnet...")
    try:
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        print(f"Failed to download wordnet: {e}")

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    print("Downloading NLTK omw-1.4...")
    try:
        nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        print(f"Failed to download omw-1.4: {e}")

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
    results = {}

    # Perplexity computation
    try:
        print("Computing perplexity...")
        ppl = compute_perplexity(predictions)
        avg_ppl = sum(ppl) / len(ppl)
        results["Perplexity (avg)"] = avg_ppl
        print(f"✅ Perplexity computed: {avg_ppl:.4f}")
    except Exception as e:
        print(f"❌ Error computing perplexity: {e}")
        results["Perplexity (avg)"] = None

    # MAUVE computation (often causes segfaults with small datasets)
    try:
        if len(predictions) >= 1000:  # Increased threshold - MAUVE needs LOTS of data
            print("Computing MAUVE...")
            mauve_score = compute_mauve(predictions, inputs)
            results["MAUVE"] = mauve_score
            print(f"✅ MAUVE computed: {mauve_score:.4f}")
        else:
            print(f"⚠️ Skipping MAUVE (dataset too small: {len(predictions)} samples, need ≥1000 for stable results)")
            results["MAUVE"] = None
    except Exception as e:
        print(f"❌ Error computing MAUVE: {e}")
        results["MAUVE"] = None

    # Self-BLEU computation
    try:
        print("Computing Self-BLEU...")
        self_bleu = compute_self_bleu(predictions)
        results["Self-BLEU"] = self_bleu
        print(f"✅ Self-BLEU computed: {self_bleu:.4f}")
    except Exception as e:
        print(f"❌ Error computing Self-BLEU: {e}")
        results["Self-BLEU"] = None

    return results


def run_zero_or_few_shot(model_name, eval_dataset, tokenizer, shots=0):
    """
    Runs zero-shot or few-shot inference on an evaluation dataset using a T5-like model.
    
    For reverse prompt engineering task:
    - input_text: Context + Response (what we give to model)
    - target_text: Instruction (what model should generate)

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
    print(f"\n🎯 Running {'zero-shot' if shots == 0 else f'{shots}-shot'} evaluation...")
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    predictions = []
    references = []
    
    # Process each example individually
    for idx, example in enumerate(eval_dataset):
        if idx % 5 == 0:
            print(f"  Processing example {idx+1}/{len(eval_dataset)}")
        
        # Build few-shot examples (if any)
        few_shot_examples = []
        if shots > 0:
            # Use different examples for few-shot (not including current one)
            shot_indices = [i for i in range(min(shots, len(eval_dataset))) if i != idx]
            for shot_idx in shot_indices[:shots]:
                shot_input = eval_dataset[shot_idx]["input_text"]
                shot_target = eval_dataset[shot_idx]["target_text"]
                few_shot_examples.append(f"Input: {shot_input}\nOutput: {shot_target}")
        
        # Build the full prompt
        if shots == 0:
            # Zero-shot: simple instruction generation prompt
            prompt = f"Generate instruction for: {example['input_text']}"
        else:
            # Few-shot: examples + current input
            examples_text = "\n\n".join(few_shot_examples)
            prompt = f"{examples_text}\n\nInput: {example['input_text']}\nOutput:"
        
        # Debug: print first few prompts
        if idx < 3:
            print(f"\n🔍 DEBUG - Example {idx+1} prompt:")
            print(f"'{prompt[:200]}{'...' if len(prompt) > 200 else ''}'")
        
        # Generate prediction
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        
        # Decode prediction (remove input tokens)
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Debug: print first few predictions
        if idx < 3:
            print(f"🎯 Prediction: '{prediction}'")
            print(f"📝 Reference: '{example['target_text']}'")
        
        predictions.append(prediction)
        references.append(example["target_text"])
        
        # Clear memory periodically
        if idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Clean up model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"✅ Generated {len(predictions)} predictions")
    return predictions, references
import torch
torch.cuda.empty_cache()

if __name__ == "__main__":
    
    # CPU Performance Warning
    if device.type == "cpu":
        print("\n⚠️  CPU PERFORMANCE WARNING ⚠️")
        print("You're running on CPU which will be significantly slower.")
        print("Expected runtime: 30-60 minutes for this debug run")
        print("Consider using smaller debug_size or running on GPU for faster results.\n")
    

    model_checkpoint = "/Users/yarinoh/PycharmProjects/ANLP/Final_Project/results/best_run/checkpoint-4389"
    print(f"Loading best model from: {model_checkpoint}")

    # Look for model files in root directory (where trainer.save_model() saves them)
    model_files = ["pytorch_model.bin", "model.safetensors"]
    has_model_in_root = any(os.path.exists(os.path.join(model_checkpoint, file)) for file in model_files)

    if has_model_in_root:
        print(f"✅ Found model in root directory: {model_checkpoint}")
    else:
        print(f"❌ No model files found. Falling back to base model...")

    print(f"📦 Actually loading model from: {model_checkpoint}")
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

    # Prepare trainer for evaluation
    eval_args = Seq2SeqTrainingArguments(
        output_dir="./results/eval",
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        gradient_checkpointing=False if device.type == "cpu" else True,  # Disable for CPU
        do_predict=True,
        report_to="none",
        dataloader_num_workers=0 if device.type == "cpu" else 2,  # Disable multiprocessing on CPU
        # ,fp16=True # Uncomment if you have a GPU and want to use mixed precision
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    # model.gradient_checkpointing_enable()

    # For debugging: use only a small subset (even smaller for CPU)
    debug_size = 100 if device.type == "cpu" else 1000  # Much smaller for CPU
    eval_dataset_debug = eval_dataset_clean.select(range(debug_size))
    
    # Clear memory before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    outputs = trainer.predict(eval_dataset_debug)

    decoded_preds = safe_decode_predictions(outputs.predictions, tokenizer)
    decoded_labels = safe_decode_predictions(outputs.label_ids, tokenizer)

    print("📊 Computing reference-based metrics...")
    metrics = compute_metrics(decoded_preds, decoded_labels)
    print(json.dumps(metrics, indent=2))
    
    # Clear memory after reference-based metrics
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("📊 Computing reference-free metrics...")
    eval_dataset_debug_raw = eval_dataset.select(range(debug_size))  # Use same debug size
    decoded_inputs = safe_decode_predictions([ex["input_ids"] for ex in eval_dataset_debug_raw], tokenizer)
    ref_free_metrics = run_reference_free_metrics(decoded_preds, decoded_inputs)
    print(json.dumps(ref_free_metrics, indent=2))
    
    # Clear memory after reference-free metrics
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Merge metrics into one file
    all_metrics = {**metrics, **ref_free_metrics}

    print("\n🚀 Running Zero-shot Evaluation...")
    small_eval_size = 20 if device.type == "cpu" else 100  # Much smaller for CPU
    small_eval = eval_dataset_raw.select(range(small_eval_size))
    zero_preds, zero_refs = run_zero_or_few_shot("t5-base", small_eval, tokenizer, shots=0)
    zero_metrics = compute_metrics(zero_preds, zero_refs)
    # zero_ref_free = run_reference_free_metrics(zero_preds, [ex["input_text"] for ex in small_eval])
    
    print("\n🚀 Running Few-shot Evaluation (e.g., 3-shot)...")
    few_preds, few_refs = run_zero_or_few_shot("t5-base", small_eval, tokenizer, shots=3)
    few_metrics = compute_metrics(few_preds, few_refs)
    # few_ref_free = run_reference_free_metrics(few_preds, [ex["input_text"] for ex in small_eval])

    # Save detailed results for paper analysis
    detailed_results = {
        "finetuned_model": {
            "reference_based": metrics,
            # "reference_free": ref_free_metrics,
            "sample_size": debug_size,
            "model_checkpoint": model_checkpoint
        },
        "zero_shot_baseline": {
            "reference_based": zero_metrics,
            # "reference_free": zero_ref_free,
            "sample_size": len(small_eval),
            "model": "t5-base"
        },
        "few_shot_baseline": {
            "reference_based": few_metrics, 
            # "reference_free": few_ref_free,
            "sample_size": len(small_eval),
            "model": "t5-base",
            "shots": 3
        }
    }
    
    # Create paper-ready summary
    paper_summary = {
        "Model Performance Comparison": {
            "Fine-tuned Model": {
                "BLEU": metrics.get("bleu", "N/A"),
                "ROUGE-1": metrics.get("rouge1", "N/A"), 
                "ROUGE-L": metrics.get("rougeL", "N/A"),
                "METEOR": metrics.get("meteor", "N/A"),
                "BARTScore": metrics.get("bart_score", "N/A"),
                # "Perplexity": ref_free_metrics.get("Perplexity (avg)", "N/A"),
                # "Self-BLEU": ref_free_metrics.get("Self-BLEU", "N/A")
            },
            "Zero-shot Baseline": {
                "BLEU": zero_metrics.get("bleu", "N/A"),
                "ROUGE-1": zero_metrics.get("rouge1", "N/A"),
                "ROUGE-L": zero_metrics.get("rougeL", "N/A"), 
                "METEOR": zero_metrics.get("meteor", "N/A"),
                "BARTScore": zero_metrics.get("bart_score", "N/A"),
                # "Perplexity": zero_ref_free.get("Perplexity (avg)", "N/A"),
                # "Self-BLEU": zero_ref_free.get("Self-BLEU", "N/A")
            },
            "Few-shot Baseline": {
                "BLEU": few_metrics.get("bleu", "N/A"),
                "ROUGE-1": few_metrics.get("rouge1", "N/A"),
                "ROUGE-L": few_metrics.get("rougeL", "N/A"),
                "METEOR": few_metrics.get("meteor", "N/A"), 
                "BARTScore": few_metrics.get("bart_score", "N/A"),
                # "Perplexity": few_ref_free.get("Perplexity (avg)", "N/A"),
                # "Self-BLEU": few_ref_free.get("Self-BLEU", "N/A")
            }
        }
    }

    with open("./results/eval/detailed_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)
        
    with open("./results/eval/paper_summary.json", "w") as f:
        json.dump(paper_summary, f, indent=2)
        
    # # Also save individual metric files for compatibility
    # with open("./results/eval/zero_shot_metrics.json", "w") as f:
    #     json.dump({**zero_metrics, **zero_ref_free}, f, indent=2)
        
    # with open("./results/eval/few_shot_metrics.json", "w") as f:
    #     json.dump({**few_metrics, **few_ref_free}, f, indent=2)
        
    print("\n📄 Paper-ready results saved to:")
    print("  - detailed_results.json (comprehensive)")
    print("  - paper_summary.json (table-ready)")

    results_summary = {
        "finetuned": all_metrics
        # "zero_shot": {**zero_metrics, **zero_ref_free},
        # "few_shot": {**few_metrics, **few_ref_free}
    }
    with open("./results/eval/summary_metrics.json", "w") as f:
        json.dump(results_summary, f, indent=2)
        
        
        