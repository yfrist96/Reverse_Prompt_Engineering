import os
import itertools
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GPT2LMHeadModel,
    GPT2TokenizerFast
)
import torch
import wandb
import numpy as np
from datetime import datetime
import json
import sys
from datasets import load_dataset, load_from_disk, DatasetDict
import mauve
import nltk
import ssl
import pandas as pd
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rouge_scorer_lib
import evaluate

# Add the cloned directory to the system path
sys.path.append('./BARTScore')
from bart_score import BARTScorer

# --------------------- Device Setup & Initialization ---------------------
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Choose your checkpoint
bart_checkpoint = "facebook/bart-large-cnn"

# Instantiate BARTScorer once
bart_scorer = BARTScorer(
    device=device,
    checkpoint=bart_checkpoint)

# CPU-specific optimizations
if device.type == "cpu":
    print("💡 CPU detected - applying CPU optimizations:")
    print("  - Reduced batch sizes")
    print("  - Disabled gradient checkpointing")
    print("  - Using smaller models where possible")

# Load tokenizer and model for our main tasks
tokenizer = AutoTokenizer.from_pretrained("t5-base")
DATA_DIR = "./data/alpaca_tokenized"


# --------------------- Device Setup & Initialization ---------------------


def is_useful(example):
    """
    Determine whether an Alpaca example is useful for reverse prompt prediction.

    Criteria for filtering out low-quality examples:
    - Output must contain more than 1 word, OR
    - Output must be longer than 5 characters and not purely numeric, OR
    - Input (context) must be non-empty

    This helps remove trivial, uninformative, or extremely short examples
    that are unlikely to help the model learn meaningful mappings.

    Args:
        example (dict): A single Alpaca dataset example with keys 'input' and 'output'.

    Returns:
        bool: True if the example is considered useful, False otherwise.
    """
    out = example["output"].strip()
    inp = example["input"].strip()
    # Filter out outputs that are too short or meaningless
    return (
            len(out.split()) > 1 or
            (len(out) > 5 and not out.isnumeric()) or
            len(inp) > 0
    )


def preprocess_fn(examples):
    """Preprocess the dataset examples for T5 training."""
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


def format_with_context(example):
    """
    Formats a dataset example for reverse prompt prediction by creating an input-output pair.

    Args:
        example (dict): A dictionary containing the keys "instruction", "input", and "output".

    Returns:
        dict: A dictionary with:
            - "input_text": A concatenation of the input and output fields, formatted as:
                  "Context: <input>\nResponse: <output>"
              If the input is empty, it defaults to just the output.
            - "target_text": The original instruction that generated the output.

    This formatting is used to train models to predict the instruction (prompt)
    given an output and optional input (i.e., reverse prompt engineering).
    """
    instruction = example["instruction"].strip()
    input_field = example["input"].strip()
    output_field = example["output"].strip()
    input_text = f"Context: {input_field}\nResponse: {output_field}" if input_field else output_field
    return {"input_text": input_text, "target_text": instruction}


def safe_decode_predictions(predictions, tokenizer, skip_special_tokens=True):
    """
    Safely decode model predictions into text using the provided tokenizer.

    This utility handles common prediction formats (e.g., logits, token IDs) and ensures
    robustness against invalid or unexpected inputs.

    Args:
        predictions (np.ndarray or tuple): Model output, which can be:
            - A 2D array of token IDs (batch_size x seq_len)
            - A 3D array of logits (batch_size x seq_len x vocab_size)
            - A tuple containing one of the above (e.g., from HuggingFace Trainer)
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer used for decoding.
        skip_special_tokens (bool): Whether to remove special tokens like <pad>, <eos>, etc. (default: True)

    Returns:
        List[str]: Decoded text outputs, one per prediction.

    Raises:
        ValueError: If predictions contain non-integer types (e.g., float logits not processed via argmax).

    Notes:
        - If logits are detected (3D array), `argmax` is applied to obtain token IDs.
        - Token IDs are clipped to valid range `[0, tokenizer.vocab_size - 1]` to avoid decode errors.
    """
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.array(predictions)

    if predictions.ndim == 3:
        print("Detected logits. Applying argmax to get token IDs.")
        predictions = np.argmax(predictions, axis=-1)

    if not np.issubdtype(predictions.dtype, np.integer):
        raise ValueError("Predictions must be integer token IDs. Got float or corrupt values.")

    predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)
    return tokenizer.batch_decode(predictions.tolist(), skip_special_tokens=skip_special_tokens)


def compute_metrics(preds, labels):
    """
    Compute a suite of standard text generation evaluation metrics:
    BLEU, ROUGE (ROUGE-1 and ROUGE-L), METEOR, and BARTScore.

    Args:
        preds (List[str]): Model predictions (generated outputs).
        labels (List[str]): Ground-truth target texts.

    Returns:
        dict: A dictionary with the following keys:
            - 'bleu': BLEU score (0-1 float or None if failed)
            - 'rouge1': ROUGE-1 F1 score
            - 'rougeL': ROUGE-L F1 score
            - 'meteor': METEOR score
            - 'bart_score': BARTScore (average across all valid pairs)

    Notes:
        - All strings are stripped before evaluation.
        - If predictions contain empty strings or invalid tokens, they may be skipped.
        - For BARTScore, if batch scoring fails, a fallback mechanism scores one pair at a time.
    """
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
        print(f"BLEU: {results['bleu']:.4f}")
    except Exception as e:
        print(f"Error computing BLEU: {e}")
        results["bleu"] = None

    # ROUGE
    try:
        rouge = evaluate.load("rouge")
        rouge_scores = rouge.compute(predictions=preds, references=labels)
        results["rouge1"] = rouge_scores["rouge1"]
        results["rougeL"] = rouge_scores["rougeL"]
        print(f"ROUGE-1: {results['rouge1']:.4f}, ROUGE-L: {results['rougeL']:.4f}")
    except Exception as e:
        print(f"Error computing ROUGE: {e}")
        results["rouge1"] = None
        results["rougeL"] = None

    # METEOR
    try:
        meteor = evaluate.load("meteor")
        results["meteor"] = meteor.compute(predictions=preds, references=labels)["meteor"]
        print(f"METEOR: {results['meteor']:.4f}")
    except Exception as e:
        print(f"Error computing METEOR: {e}")
        results["meteor"] = None

    # BARTSCORE
    try:
        print("Computing BARTScore...")

        # Filter out pairs with empty strings
        valid_pairs = [(p, l) for p, l in zip(preds, labels) if p.strip() and l.strip()]
        if not valid_pairs:
            print("No valid prediction-label pairs for BARTScore")
            results["bart_score"] = None
        else:
            valid_preds, valid_labels = zip(*valid_pairs)
            valid_preds = list(valid_preds)
            valid_labels = list(valid_labels)

            print(f"  Processing {len(valid_pairs)} valid pairs...")

            # 💡 Let BARTScorer handle the batching internally.
            # It's more efficient and the intended usage.
            bart_scores = bart_scorer.score(valid_preds, valid_labels, batch_size=16)  # Set batch_size here

            avg_bart_score = np.mean(bart_scores)
            results["bart_score"] = avg_bart_score
            print(f"BARTScore: {avg_bart_score:.4f}")

    except Exception as e:
        print(f"An error occurred during initial BARTScore calculation: {e}")
        print("Attempting to score one-by-one to find and skip problematic pairs...")

        # Fallback: score one by one to isolate the issue
        scores = []
        problematic_count = 0
        for i, (pred, label) in enumerate(valid_pairs):
            try:
                # Score a single pair
                score = bart_scorer.score([pred], [label])
                scores.append(score[0])
            except Exception as single_e:
                problematic_count += 1
                # You can uncomment the line below to see exactly which pair failed
                # print(f"  ⚠️ Skipping pair {i} due to error: {single_e}\n    PRED: '{pred}'\n    LABEL: '{label}'")
                continue  # Skip this pair and continue

        if scores:
            avg_bart_score = np.mean(scores)
            results["bart_score"] = avg_bart_score
            print(
                f"BARTScore (fallback): {avg_bart_score:.4f} (calculated from {len(scores)} pairs, skipped {problematic_count})")
        else:
            print("BARTScore could not be computed for any pair.")
            results["bart_score"] = None

    return results


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
                print(f"  Processing text {i + 1}/{len(texts)}")

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

    # Filter out inf values for downstream averaging
    valid_perplexities = [x for x in perplexities if np.isfinite(x)]
    return valid_perplexities


# TODO - pip install mauve-text
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
    result = mauve.compute_mauve(p_text=p_texts, q_text=q_texts, device_id=device_id)
    return result.mauve


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
        references = generations[:i] + generations[i + 1:]
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
        print(f"Perplexity computed: {avg_ppl:.4f}")
    except Exception as e:
        print(f"Error computing perplexity: {e}")
        results["Perplexity (avg)"] = None

    # MAUVE computation (often causes segfaults with small datasets)
    try:
        if len(predictions) >= 1000:  # MAUVE needs LOTS of data
            print("Computing MAUVE...")
            mauve_score = compute_mauve(predictions, inputs)
            results["MAUVE"] = mauve_score
            print(f"MAUVE computed: {mauve_score:.4f}")
        else:
            print(f"Skipping MAUVE (dataset too small: {len(predictions)} samples, need ≥1000 for stable results)")
            results["MAUVE"] = None
    except Exception as e:
        print(f"Error computing MAUVE: {e}")
        results["MAUVE"] = None

    # Self-BLEU computation
    try:
        print("Computing Self-BLEU...")
        self_bleu = compute_self_bleu(predictions)
        results["Self-BLEU"] = self_bleu
        print(f"Self-BLEU computed: {self_bleu:.4f}")
    except Exception as e:
        print(f"Error computing Self-BLEU: {e}")
        results["Self-BLEU"] = None

    return results


def run_zero_or_few_shot(model_name, eval_dataset, tokenizer, shots=0, demo_examples=None):
    """
    Clean implementation of zero-shot and few-shot evaluation for reverse prompt engineering.

    Task: Given a response/output, generate the instruction/question that would produce it.

    Data format:
    - input_text: "Context: X\nResponse: Y" or just "Y" (the answer/response)
    - target_text: "What is the instruction?" (the question we want to generate)

    Args:
        model_name (str): HuggingFace model (e.g., "google/flan-t5-base")
        eval_dataset: Dataset with "input_text" and "target_text" fields
        tokenizer: Compatible tokenizer (can be None, will load fresh)
        shots (int): Number of demonstration examples (0 = zero-shot)

    Returns:
        Tuple[List[str], List[str]]: (predictions, references)
    """
    print(f"\nStarting {'Zero-shot' if shots == 0 else f'{shots}-shot'} Evaluation")
    print(f"Dataset size: {len(eval_dataset)} examples")
    print(f"Model: {model_name}")

    # Load model and tokenizer fresh
    print("Loading model and tokenizer...")
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model = model.to(device)
        model.eval()

        fresh_tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return [], []

    # Collect demonstration examples for few-shot (if needed)
    # demo_examples = []
    if shots > 0:
        if demo_examples is None:
            raise ValueError("For few-shot, demo_examples must be provided.")
        print(f"Using {len(demo_examples)} fixed demonstration examples.")

    # Process each example
    predictions = []
    references = []
    prompts = []

    print("Starting inference...")
    for idx, example in enumerate(tqdm(eval_dataset, desc=f"Running {model_name}")):
        # Progress reporting
        if idx % 5 == 0 or idx < 5:
            print(f"  Processing {idx + 1}/{len(eval_dataset)}")

        # Build the prompt
        prompt = _build_prompt(example, demo_examples, shots)
        prompts.append(prompt)

        # Show first few prompts for debugging
        if idx < 3 or (shots > 0 and idx == shots):
            print(f"\n🔍 Example {idx + 1} Prompt:")
            print(f"'{prompt[:150]}{'...' if len(prompt) > 150 else ''}'")

        # Generate prediction
        try:
            prediction = _generate_prediction(model, fresh_tokenizer, prompt)

            # Debug first few predictions
            if idx < 3 or (shots > 0 and idx == shots):
                print(f"Prediction: '{prediction}'")
                print(f"Target: '{example['target_text']}'")
                print(f"Valid: {bool(prediction.strip())}")

        except Exception as e:
            print(f"Error generating prediction for example {idx + 1}: {e}")
            prediction = ""

        predictions.append(prediction)
        references.append(example["target_text"])

        # Memory cleanup
        if idx % 10 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Final cleanup
    del model
    del fresh_tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary
    valid_preds = sum(1 for p in predictions if p.strip())
    print(f"\nEvaluation Summary:")
    print(f"  Total examples: {len(predictions)}")
    print(f"  Valid predictions: {valid_preds}/{len(predictions)} ({valid_preds / len(predictions) * 100:.1f}%)")
    print(f"  Empty predictions: {len(predictions) - valid_preds}")

    return predictions, references, prompts


def _build_prompt(example, demo_examples, shots):
    """Build the prompt for zero-shot or few-shot inference."""

    if shots == 0:
        # Zero-shot: direct instruction format
        # return f"Generate a question for this answer: {example['input_text']}"
        return f"Given this output, what prompt likely generated it?\n\nOutput: {example['input_text']}\nPrompt:"


    else:
        # Few-shot: demonstrations + current example
        prompt_parts = []

        # Add demonstration examples
        print(f"Using {len(demo_examples)} demonstration examples")
        for demo in demo_examples:
            prompt_parts.append(f"Answer: {demo['input']}")
            prompt_parts.append(f"Question: {demo['output']}")
            prompt_parts.append("")  # Empty line between examples

        # Add current example
        prompt_parts.append(f"Answer: {example['input_text']}")
        prompt_parts.append("Question:")

        return "\n".join(prompt_parts)


def _generate_prediction(model, tokenizer, prompt):
    """Generate a single prediction from the model."""

    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    # Generate - T5 is encoder-decoder, so it generates full response, not continuation
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            max_length=128,  # Increased max length slightly
            num_beams=4,  # Use beam search
            early_stopping=True,  # Stop when beams converge
        )

    # For T5, the output is the complete generated sequence (not input + generated)
    # So we just decode the full output
    # In safe_decode_predictions or _generate_prediction

    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction.strip()


def run_full_analysis(model_name, decoded_inputs, decoded_preds, decoded_labels, output_filename, model_prompts=None):
    """
    Computes per-sample metrics (BLEU, ROUGE, METEOR, BARTScore), aggregates them,
    and saves a detailed analysis to a CSV file.

    Args:
        model_name (str): A name for the model being evaluated (e.g., "fine-tuned").
        decoded_inputs (List[str]): The original input texts.
        decoded_preds (List[str]): The model's predicted prompts.
        decoded_labels (List[str]): The ground truth prompts.
        output_filename (str): Path to save the output CSV file.

    Returns:
        Tuple[dict, pd.DataFrame]: A tuple containing:
            - A dictionary of the aggregated metrics.
            - A pandas DataFrame with the detailed per-sample results.
    """
    print(f"Running full per-sample analysis for '{model_name}' model...")
    results_data = []

    # --- 1. Batch-calculate BARTScore for efficiency ---
    print("  Calculating BARTScore for all samples...")
    # Filter out pairs with empty strings, which BARTScorer can't handle
    valid_indices = [
        i for i, (p, l) in enumerate(zip(decoded_preds, decoded_labels)) if p.strip() and l.strip()
    ]
    valid_preds = [decoded_preds[i] for i in valid_indices]
    valid_labels = [decoded_labels[i] for i in valid_indices]

    # Initialize a list to hold all scores, with 0.0 for invalid pairs
    all_bart_scores = [0.0] * len(decoded_preds)
    if valid_preds:
        try:
            # The global `bart_scorer` must be initialized before this function is called
            bart_scores_list = bart_scorer.score(valid_preds, valid_labels, batch_size=16)
            # Place calculated scores back into their original positions
            for i, score_idx in enumerate(valid_indices):
                all_bart_scores[score_idx] = bart_scores_list[i]
        except Exception as e:
            print(f"  BARTScore batch calculation failed: {e}. Scores will be 0.")

    # --- 2. Initialize scorers for per-sample metrics ---
    # This is more efficient than re-initializing them inside the loop
    rouge_calc = rouge_scorer_lib.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    meteor_calc = evaluate.load('meteor')
    smoothing_fn = SmoothingFunction().method1  # Smoothing for BLEU scores

    # If prompts aren't provided, create a list of placeholders
    if model_prompts is None:
        model_prompts = ["N/A"] * len(decoded_preds)
    # --- 3. Loop through each sample and calculate metrics ---
    print("  Calculating per-sample metrics (BLEU, ROUGE, METEOR)...")
    for i, (inp, pred, label, prompt_text) in enumerate(
            tqdm(zip(decoded_inputs, decoded_preds, decoded_labels, model_prompts), total=len(decoded_preds),
                 desc="Analyzing samples")):
        pred = decoded_preds[i].strip()
        label = decoded_labels[i].strip()

        # Some metrics require non-empty strings
        if not pred or not label:
            continue

        # BLEU with smoothing
        bleu_score = sentence_bleu([label.split()], pred.split(), smoothing_function=smoothing_fn)

        # ROUGE
        rouge_results = rouge_calc.score(label, pred)

        # METEOR
        meteor_score = meteor_calc.compute(predictions=[pred], references=[label])['meteor']

        results_data.append({
            'model_prompt': prompt_text,
            'input_context': decoded_inputs[i],
            'predicted_prompt': pred,
            'actual_prompt': label,
            'bleu': bleu_score,
            'rouge1': rouge_results['rouge1'].fmeasure,
            'rougeL': rouge_results['rougeL'].fmeasure,
            'meteor': meteor_score,
            'bart_score': all_bart_scores[i]
        })

    if not results_data:
        print("No valid results were generated to save.")
        return {}, None

    # --- 4. Convert to DataFrame, calculate aggregates, and save ---
    df = pd.DataFrame(results_data)

    # Calculate aggregate metrics by averaging the per-sample scores
    aggregate_metrics = {
        'bleu': df['bleu'].mean(),
        'rouge1': df['rouge1'].mean(),
        'rougeL': df['rougeL'].mean(),
        'meteor': df['meteor'].mean(),
        # For BARTScore, only average the non-zero scores
        'bart_score': df[df['bart_score'] > 0]['bart_score'].mean() if not df[df['bart_score'] > 0].empty else 0.0
    }

    print("\nAggregate Metrics (from per-sample analysis):")
    print(json.dumps(aggregate_metrics, indent=2))

    # Save the detailed DataFrame to a CSV file
    df.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"Detailed analysis saved to '{output_filename}'")

    return aggregate_metrics, df


# -------------------------------------- Dataset Preparation --------------------------------------
#
# 1) Load & format the RAW Alpaca once
#
raw = load_dataset("tatsu-lab/alpaca", split="train")
filtered_raw = raw.filter(is_useful)

formatted = filtered_raw.map(format_with_context,
                             remove_columns=["instruction", "input", "output"])

print(f"Original size: {len(raw)}, Filtered size: {len(filtered_raw)}")

#
# 2) Split the _formatted_ into train/test
#
split_formatted = formatted.train_test_split(test_size=0.1, seed=42)
train_formatted = split_formatted["train"]  # has input_text & target_text
eval_formatted = split_formatted["test"]  # has input_text & target_text

#
# 3) Tokenize _those same_ splits (and drop the text columns)
#
train_tokenized = train_formatted.map(
    preprocess_fn, batched=True,
    remove_columns=["input_text", "target_text"]
)
eval_tokenized = eval_formatted.map(
    preprocess_fn, batched=True,
    remove_columns=["input_text", "target_text"]
)

tokenized_dataset = DatasetDict({
    "train": train_tokenized,
    "test": eval_tokenized
})

# cache to disk so you don’t pay the tokenization cost again
if not os.path.isdir(DATA_DIR):
    print(f"Saving tokenized dataset to {DATA_DIR}")
    tokenized_dataset.save_to_disk(DATA_DIR)
else:
    print("Loading tokenized dataset from disk…")
    tokenized_dataset = load_from_disk(DATA_DIR)

#
# 4) Prepare for Trainer and for zero/few‐shot separately
#
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]

# this one is only for Trainer.predict:
eval_dataset_clean = eval_dataset.remove_columns(
    [c for c in eval_dataset.column_names
     if c not in ["input_ids", "attention_mask", "labels"]]
)

# this raw/textual one is for your zero/few‐shot routine:
eval_dataset_raw = eval_formatted

print(f"TRAIN size: {len(train_dataset)}")
print(f"EVAL (tokenized) size: {len(eval_dataset)}")
print(f"EVAL (raw/text) size: {len(eval_dataset_raw)}")
print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
# -------------------------------------- Dataset Preparation --------------------------------------


# -------------------------------------- NLTK setup --------------------------------------
# TODO - pip install nltk
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

# -------------------------------------- NLTK setup --------------------------------------


torch.cuda.empty_cache()

if __name__ == "__main__":

    # 1. Select 3 diverse, high-quality examples from the TRAINING data
    # This avoids any data leakage from the evaluation set.
    fixed_demo_examples = [{"input": eval_dataset_raw[i]["input_text"], "output": eval_dataset_raw[i]["target_text"]}
                           for i in [0, 10, 30]]

    # CPU Performance Warning
    if device.type == "cpu":
        print("\n  CPU PERFORMANCE WARNING")
        print("You're running on CPU which will be significantly slower.")
        print("Expected runtime: 30-60 minutes for this debug run")
        print("Consider using smaller debug_size or running on GPU for faster results.\n")

    model_checkpoint = "/Users/yarinoh/PycharmProjects/ANLP/Final_Project/results/best_run"
    print(f"Loading best model from: {model_checkpoint}")

    # Look for model files in root directory (where trainer.save_model() saves them)
    model_files = ["pytorch_model.bin", "model.safetensors"]
    has_model_in_root = any(os.path.exists(os.path.join(model_checkpoint, file)) for file in model_files)

    if has_model_in_root:
        print(f"Found model in root directory: {model_checkpoint}")
    else:
        print(f"No model files found. Falling back to base model...")

    print(f"Actually loading model from: {model_checkpoint}")
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

    # For debugging: use only a small subset (even smaller for CPU)
    debug_size = 500 if device.type == "cpu" else 1000  # Much smaller for CPU
    # eval_dataset_debug = eval_dataset_clean.select(range(debug_size))

    # Clear memory before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    outputs = trainer.predict(eval_dataset_clean)

    decoded_preds = safe_decode_predictions(outputs.predictions, tokenizer)
    decoded_labels = safe_decode_predictions(outputs.label_ids, tokenizer)

    # --- Analysis for Fine-Tuned Model ---
    print("\n" + "=" * 50)
    print("RUNNING ANALYSIS FOR FINE-TUNED MODEL")
    print("=" * 50)
    # Get predictions using the trainer
    debug_size = 500 if device.type == "cpu" else 1000
    # eval_dataset_debug = eval_dataset_clean.select(range(debug_size))
    outputs = trainer.predict(eval_dataset_clean)

    # Decode predictions, labels, and inputs
    decoded_preds = safe_decode_predictions(outputs.predictions, tokenizer)
    decoded_labels = safe_decode_predictions(outputs.label_ids, tokenizer)
    # eval_dataset_debug_raw = eval_dataset_raw.select(range(debug_size))
    decoded_inputs = [ex['input_text'] for ex in eval_dataset_raw]

    finetuned_metrics, _ = run_full_analysis(
        model_name="fine-tuned",
        decoded_inputs=decoded_inputs,
        decoded_preds=decoded_preds,
        decoded_labels=decoded_labels,
        output_filename="./results/eval/finetuned_detailed_analysis.csv"
    )

    # --- Analysis for Zero-Shot Baseline ---
    print("\n" + "=" * 50)
    print("RUNNING ANALYSIS FOR ZERO-SHOT BASELINE")
    print("=" * 50)
    small_eval_size = 500 if device.type == "cpu" else 1000
    small_eval = eval_dataset_raw.select(range(small_eval_size))
    zero_preds, zero_refs, zero_prompts = run_zero_or_few_shot("google/flan-t5-base", small_eval, tokenizer, shots=0)

    zero_metrics, _ = run_full_analysis(
        model_name="zero-shot",
        decoded_inputs=[ex["input_text"] for ex in small_eval],
        decoded_preds=zero_preds,
        decoded_labels=zero_refs,
        output_filename="./results/eval/zero_shot_detailed_analysis.csv",
        model_prompts=zero_prompts
    )

    # --- Analysis for Few-Shot Baseline ---
    print("\n" + "=" * 50)
    print("RUNNING ANALYSIS FOR FEW-SHOT BASELINE")
    print("=" * 50)
    few_preds, few_refs, few_prompts = run_zero_or_few_shot(
        "google/flan-t5-base",
        small_eval,
        tokenizer,
        shots=3,
        demo_examples=fixed_demo_examples
    )

    few_metrics, _ = run_full_analysis(
        model_name="few-shot",
        decoded_inputs=[ex["input_text"] for ex in small_eval],
        decoded_preds=few_preds,
        decoded_labels=few_refs,
        output_filename="./results/eval/few_shot_detailed_analysis.csv",
        model_prompts=few_prompts
    )

    # --- (You can now update your paper_summary.json creation using the new metric dicts) ---

    # --- Re-run Reference-Free Metrics for Summary Files ---
    # The new analysis function focuses on per-sample reference-based metrics.
    # We can run your original reference-free function to get those aggregates back.

    print("\n" + "=" * 50)
    print("GATHERING REFERENCE-FREE METRICS FOR SUMMARIES")
    print("=" * 50)

    # For fine-tuned model
    print("  Calculating reference-free metrics for Fine-Tuned model...")
    finetuned_ref_free_metrics = run_reference_free_metrics(decoded_preds, decoded_inputs)
    all_finetuned_metrics = {**finetuned_metrics, **finetuned_ref_free_metrics}

    # For zero-shot model
    print("\n  Calculating reference-free metrics for Zero-Shot model...")
    zero_ref_free_metrics = run_reference_free_metrics(zero_preds, [ex["input_text"] for ex in small_eval])
    all_zero_shot_metrics = {**zero_metrics, **zero_ref_free_metrics}

    # For few-shot model
    print("\n  Calculating reference-free metrics for Few-Shot model...")
    few_ref_free_metrics = run_reference_free_metrics(few_preds, [ex["input_text"] for ex in small_eval])
    all_few_shot_metrics = {**few_metrics, **few_ref_free_metrics}

    # --- Create and Save Final JSON Summary Files ---
    print("\n" + "=" * 50)
    print("CREATING AND SAVING FINAL JSON SUMMARIES")
    print("=" * 50)

    # 1. Create the detailed summary dictionary
    detailed_results = {
        "finetuned_model": {
            "metrics": all_finetuned_metrics,
            "sample_size": len(eval_dataset_raw),
            "model_checkpoint": model_checkpoint
        },
        "zero_shot_baseline": {
            "metrics": all_zero_shot_metrics,
            "sample_size": len(small_eval),
            "model": "google/flan-t5-base"
        },
        "few_shot_baseline": {
            "metrics": all_few_shot_metrics,
            "sample_size": len(small_eval),
            "model": "google/flan-t5-base",
            "shots": 3
        }
    }


    # Helper function to format the metrics for the paper summary
    def get_metrics_for_summary(metrics_dict):
        return {
            "BLEU": metrics_dict.get("bleu"),
            "ROUGE-1": metrics_dict.get("rouge1"),
            "ROUGE-L": metrics_dict.get("rougeL"),
            "METEOR": metrics_dict.get("meteor"),
            "BARTScore": metrics_dict.get("bart_score"),
            "Perplexity": metrics_dict.get("Perplexity (avg)"),
            "MAUVE": metrics_dict.get("MAUVE"),
            "Self-BLEU": metrics_dict.get("Self-BLEU")
        }


    # 2. Create the paper-ready summary table
    paper_summary = {
        "Model Performance Comparison": {
            "Fine-tuned Model": get_metrics_for_summary(all_finetuned_metrics),
            "Zero-shot Baseline": get_metrics_for_summary(all_zero_shot_metrics),
            "Few-shot Baseline": get_metrics_for_summary(all_few_shot_metrics)
        }
    }


    # Format numbers to 4 decimal places for cleaner output
    def format_dict_numbers(d):
        if isinstance(d, dict):
            return {k: format_dict_numbers(v) for k, v in d.items()}
        elif isinstance(d, float):
            return f"{d:.4f}"
        return d


    # 3. Save the files
    try:
        with open("./results/eval/detailed_results.json", "w") as f:
            json.dump(detailed_results, f, indent=2)

        with open("./results/eval/paper_summary.json", "w") as f:
            # Save the formatted version for easy copy-pasting into a paper
            json.dump(format_dict_numbers(paper_summary), f, indent=2)

        print("Successfully saved summary files:")
        print("   - ./results/eval/detailed_results.json (raw numbers)")
        print("   - ./results/eval/paper_summary.json (formatted for tables)")

    except Exception as e:
        print(f"Error saving JSON files: {e}")