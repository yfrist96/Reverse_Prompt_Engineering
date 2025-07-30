# 🔁 Reverse Prompt Prediction with T5

This project explores the **inverse task of language generation**:  
Given a model’s output (e.g., a completion), can we reconstruct the original prompt that generated it?

Reverse prompt prediction enhances interpretability.

We fine-tune a T5-based model on the [Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca) to reconstruct prompts from completions. The project also compares **zero-shot** and **few-shot** performance baselines.

## 📁 Project Structure

```
.
├── eval/                          # Evaluation outputs and results
│   ├── few_shot_detailed_analysis.csv
│   ├── finetuned_detailed_analysis.csv
│   ├── paper_summary.json
│   └── zero_shot_detailed_analysis.csv
├── plots/                         # Generated visualizations (PNGs)
├── eval.py                        # Evaluation logic
├── plot.py                        # Plotting scripts
├── train.py                       # Model training script
├── requirements.txt
└── README.md
```

## 🚀 Quickstart

### 1. Clone this repository

```bash
git clone https://github.com/YarinOhayon/Reverse_Prompt_Engineering.git
cd reverse-prompt-prediction
```

### 2. Set up a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install core dependencies

```bash
pip install -r requirements.txt
```

## 📦 Additional Dependencies for Evaluation

To compute advanced evaluation metrics like **BARTScore**, **MAUVE**, and **METEOR**, run:

```bash
# BARTScore (cloned from original repo)
git clone https://github.com/neulab/BARTScore.git
cd BARTScore
pip install -e .
cd ..

# MAUVE
pip install mauve-text

# NLTK (for METEOR)
pip install nltk
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
```

## 🏋️‍♀️ Training

Fine-tune the model using:

```bash
python train.py
```

## 📊 Evaluation

Evaluate the model with:

```bash
python eval.py
```

Detailed results will be saved under the `eval/` folder.

## 📉 Visualization

Generate plots for metric comparison and error analysis:

```bash
python plot.py
```

This will save figures to the `plots/` folder, including:
- Metric comparison across models
- Metric correlation heatmap
- Input length vs. score error analysis

## 🧪 Evaluation Metrics

Metrics computed per model:
- BLEU
- ROUGE-1, ROUGE-L
- METEOR
- BARTScore
- MAUVE
- Self-BLEU
- Perplexity

## 📁 Example Output

- `eval/few_shot_detailed_analysis.csv`
- `eval/finetuned_detailed_analysis.csv`
- `eval/zero_shot_detailed_analysis.csv`
- Visualizations in `plots/`

## 🤝 Citation

If you use or reference this work, please cite it as:

> Ohayon, Y. (2025). *Reverse Prompt Prediction for Interpretability in Language Models*.

## 🛠️ TODO

- [ ] Add support for larger models (e.g., T5-large)
- [ ] Add ablation studies on prompt complexity
- [ ] Integrate attention visualization

## 📬 Contact

For questions or collaboration: **@Yarin Ohayon**
