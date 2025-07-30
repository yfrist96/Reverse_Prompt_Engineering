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

# --------------------- Device Setup & Initialization ---------------------
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-base")
# --------------------- Device Setup & Initialization ---------------------


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
        print("Detected logits. Applying argmax to get token IDs.")
        predictions = np.argmax(predictions, axis=-1)

    if not np.issubdtype(predictions.dtype, np.integer):
        raise ValueError("Predictions must be integer token IDs. Got float or corrupt values.")

    predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)
    return tokenizer.batch_decode(predictions.tolist(), skip_special_tokens=skip_special_tokens)

def train_single_config(config, run_name):
    """Train a single model configuration and return results."""
    
    # Initialize wandb for this run
    wandb.init(
        project="reverse-prompt-prediction-hyperparameter-tuning",
        name=run_name,
        config=config,
        reinit=True
    )
    
    print(f"\nStarting training for: {run_name}")
    print(f"Config: {config}")
    
    # Create model
    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Create output directory for this run
    output_dir = f"./results/{run_name}"
    
    # Training arguments with current config
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        predict_with_generate=True,
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        lr_scheduler_type=config["lr_scheduler_type"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_epochs"],
        weight_decay=config["weight_decay"],
        logging_dir=f"./logs/{run_name}",
        save_total_limit=1,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=True if device.type == "cuda" else False,
        fp16=True if device.type == "cuda" else False,  # Use mixed precision on GPU
    )
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Train the model
    train_result = trainer.train()
    
    # Evaluate on test set
    print(f"Running evaluation for {run_name}...")
    eval_results = trainer.evaluate()
    
    # Run predictions for sample analysis
    print(f"Generating predictions for {run_name}...")
    outputs = trainer.predict(eval_dataset_clean)
    
    # Decode predictions and labels
    decoded_preds = safe_decode_predictions(outputs.predictions, tokenizer)
    decoded_labels = safe_decode_predictions(outputs.label_ids, tokenizer)
    
    # Log sample predictions to wandb
    sample_predictions = []
    for i in range(min(10, len(decoded_preds))):
        sample_predictions.append({
            "predicted": decoded_preds[i],
            "actual": decoded_labels[i]
        })
    
    # Log additional metrics to wandb
    wandb.log({
        "final_train_loss": train_result.training_loss,
        "final_eval_loss": eval_results["eval_loss"],
        "sample_predictions": wandb.Table(
            columns=["predicted", "actual"],
            data=[[pred["predicted"], pred["actual"]] for pred in sample_predictions]
        )
    })
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    trainer.save_model(output_dir)
    
    # Save config and results
    results = {
        "config": config,
        "train_loss": train_result.training_loss,
        "eval_loss": eval_results["eval_loss"],
        "run_name": run_name
    }
    
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Completed training for {run_name}")
    print(f"Train Loss: {train_result.training_loss:.4f}, Eval Loss: {eval_results['eval_loss']:.4f}")
    
    # Clean up GPU memory
    del model, trainer
    torch.cuda.empty_cache() if device.type == "cuda" else None
    
    wandb.finish()
    
    return results

def run_hyperparameter_tuning():
    """Run hyperparameter tuning with multiple configurations."""
    
    # Define hyperparameter grid
    hyperparameter_grid = {
        "learning_rate": [1e-5, 3e-5, 5e-5],
        "batch_size": [4, 8] if device.type == "cuda" else [2, 4],
        "num_epochs": [2, 3],
        "weight_decay": [0.01, 0.1],
        "warmup_steps": [100, 500],
        "lr_scheduler_type": ["linear", "cosine"],
        "gradient_accumulation_steps": [1, 2]
    }
    
    # Generate all combinations
    keys = hyperparameter_grid.keys()
    values = hyperparameter_grid.values()
    combinations = list(itertools.product(*values))
    
    print(f"Total configurations to test: {len(combinations)}")
    
    # Limit combinations if too many (optional)
    max_configs = 20  # Adjust based on your resources
    if len(combinations) > max_configs:
        print(f"Too many combinations ({len(combinations)}). Randomly sampling {max_configs}.")
        import random
        random.seed(42)
        combinations = random.sample(combinations, max_configs)
    
    all_results = []
    
    
    for i, combination in enumerate(combinations):
        config = dict(zip(keys, combination))
        run_name = f"run_{i+1:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            result = train_single_config(config, run_name)
            all_results.append(result)
            
        except Exception as e:
            print(f"Error in {run_name}: {str(e)}")
            continue
    
    # Find best configuration
    if all_results:
        best_result = min(all_results, key=lambda x: x["eval_loss"])
        print(f"\nBEST CONFIGURATION:")
        print(f"Eval Loss: {best_result['eval_loss']:.4f}")
        print(f"Config: {best_result['config']}")
        print(f"Run Name: {best_result['run_name']}")
        
        # Save best results
        with open("./results/best_hyperparameters.json", "w") as f:
            json.dump(best_result, f, indent=2)
        
        # Save all results
        with open("./results/all_hyperparameter_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
    
    return all_results

# -------------------------------------- Dataset Preparation --------------------------------------
# Load and prepare dataset (do this once)
print("Loading and preprocessing dataset...")
dataset = load_dataset("tatsu-lab/alpaca")
formatted_dataset = dataset["train"].map(format_with_context)
tokenized_dataset = formatted_dataset.map(preprocess_function, batched=True)
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Prepare eval dataset for prediction
eval_dataset_clean = eval_dataset.remove_columns(
    [col for col in eval_dataset.column_names if col not in ["input_ids", "attention_mask", "labels"]]
)

print(f"Dataset sizes - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
# -------------------------------------- Dataset Preparation --------------------------------------
if __name__ == "__main__":
    # Create results directory
    os.makedirs("./results_best", exist_ok=True)
    os.makedirs("./logs__", exist_ok=True)
    
    print("Starting hyperparameter tuning...")

    # result =train_single_config({
    #     "learning_rate": 5e-5,
    #     "batch_size": 4,
    #     "num_epochs": 3,
    #     "weight_decay": 0.01,
    #     "warmup_steps": 500,
    #     "lr_scheduler_type": "cosine",
    #     "gradient_accumulation_steps": 2
    # }, "best_run")
    # `results = run_hyperparameter_tuning()` is calling the function `run_hyperparameter_tuning()`
    # which is responsible for running hyperparameter tuning with multiple configurations.
    results = run_hyperparameter_tuning()
    
    print(f"✅ Hyperparameter tuning completed! Tested {len(results)} configurations.")
