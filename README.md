# 🔁 Reverse Prompt Prediction with T5

This project explores the **inverse task of language generation**:  
Given a model’s output (e.g., a completion), can we reconstruct the original prompt that generated it?

Reverse prompt prediction enhances interpretability.

We fine-tune a T5-based model on the [Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca) to reconstruct prompts from completions. The project also compares **zero-shot** and **few-shot** performance baselines.

📄 For more details, see our full paper [here](./From_Output_to_Input.pdf).

## 📁 Project Structure

```
.
├── eval/                                # Evaluation CSVs and analysis
│   ├── few_shot_detailed_analysis.csv
│   ├── finetuned_detailed_analysis.csv
│   ├── zero_shot_detailed_analysis.csv
│   └── paper_summary.json
│   └── plots/                               # Metric visualizations
│       ├── few-shot_bartscore_boxplot.png
│       ├── few-shot_bartscore_vs_length.png
│       ├── few-shot_metric_correlation_heatmap.png
│       ├── finetuned_bartscore_boxplot.png
│       ├── finetuned_bartscore_vs_length.png
│       ├── finetuned_metric_correlation_heatmap.png
│       ├── metric_comparison_across_models.png
│       ├── zero-shot_bartscore_boxplot.png
│       ├── zero-shot_bartscore_vs_length.png
│       └── zero-shot_metric_correlation_heatmap.png
├── README.md
├── From_Output_to_Input.pdf
├── requirements.txt
├── best_hyperparameters.json
├── eval.py
├── plot.py
├── train.py
└── best_model.zip (hosted on Google Drive)
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

Fine-tune the model with:

```bash
python train.py
```

## 📊 Evaluation

Run evaluation on each model variant using:

```bash
python eval.py
```

Metrics are computed on a held-out test set. The following are supported:
- BLEU
- ROUGE (1, L)
- METEOR
- BARTScore
- MAUVE
- Self-BLEU
- Perplexity

## 📈 Visualization

Run plotting with:

```bash
python plot.py
```

The project generates:
- Metric comparison bar plots
- Metric correlation heatmaps
- Input length vs score error analysis

## 🤖 Pretrained Model

The fine-tuned [model](https://drive.google.com/drive/folders/1mIbl2XGp1J6JDURod3OMl9DPxGVRk5VK?usp=sharing)
 is available as a zip archive (`best_run.zip`) hosted on Google Drive.

## 📁 Example Output

See sample visualizations in `plots/`:
- `metric_comparison_across_models.png`
- `*_bartscore_boxplot.png`
- `*_bartscore_vs_length.png`
- `*_metric_correlation_heatmap.png`

## 🤝 Citation

If you use or reference this work, please cite it as:

> Busbib, D., Frist, Y., & Ohayon, Y. (2025). From Output to Input: Modeling the Reverse Prompt Engineering Problem.

## 📬 Contact

For questions or collaboration please contact us:   **david.busbib@mail.huji.ac.il**, **yehuda.frist@mail.huji.ac.il** ,**yarin.ohayon@mail.huji.ac.il**
