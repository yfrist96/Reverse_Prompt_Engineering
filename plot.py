import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# Load the finetuned CSV
df_finetuned = pd.read_csv('results/eval/finetuned_detailed_analysis.csv')
df_finetuned['model'] = 'finetuned'  # Tag for the model type
# Load the zero-shot CSV
df_zero_shot = pd.read_csv('results/eval/zero_shot_detailed_analysis.csv')
df_zero_shot['model'] = 'zero-shot'  # Tag for the model type
# Load the few-shot CSV
df_few_shot = pd.read_csv('results/eval/few_shot_detailed_analysis.csv')
df_few_shot['model'] = 'few-shot'  # Tag for
os.makedirs('results/eval/plots', exist_ok=True)  # Ensure the directory exists

# Double default font size globally
mpl.rcParams.update({
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'font.size': 18
})
def plot_performance_distribution(df, name):
    # Plot box plot of BARTScore
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x='model', y='bart_score', palette='pastel')

    plt.title(f'{name} BARTScore Distribution')
    plt.ylabel('BARTScore')
    plt.xlabel('')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'results/eval/plots/{name.lower()}_bartscore_boxplot.png')
    plt.show()

def plot_metric_correlation_heatmap(df, name):
    # Compute the correlation matrix
    metric_cols = ['bleu', 'rouge1', 'rougeL', 'meteor', 'bart_score']
    df_metrics = df[metric_cols]
    corr = df_metrics.corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title(f'{name} Metric Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f'results/eval/plots/{name.lower()}_metric_correlation_heatmap.png')
    plt.show()

def plot_error_analysis(df, name):
    # Add input length column (in tokens or characters)
    df['input_length'] = df['input_context'].apply(lambda x: len(x.split()))  # token-based
    # OR: use len(x) for character length

    # Scatter plot: BARTScore vs input length
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='input_length', y='bart_score', alpha=0.5)
    sns.regplot(data=df, x='input_length', y='bart_score', scatter=False, color='red', ci=None)

    plt.title(f'BARTScore vs Input Length ({name})')
    plt.xlabel('Input Length (Tokens)')
    plt.ylabel('BARTScore')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/eval/plots/{name.lower()}_bartscore_vs_length.png')
    plt.show()

def plot_metric_comparison_bar_plot(df, name):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='model', y='bart_score', palette='pastel')
    plt.title(f'{name} BARTScore Comparison')
    plt.ylabel('BARTScore')
    plt.xlabel('Model')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'results/eval/plots/{name.lower()}_bartscore_comparison.png')
    plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_metrics_on_shared_subset():
    # Load each model's detailed results
    df_ft = pd.read_csv('results/eval/finetuned_detailed_analysis.csv')
    df_zs = pd.read_csv('results/eval/zero_shot_detailed_analysis.csv')
    df_fs = pd.read_csv('results/eval/few_shot_detailed_analysis.csv')

    # Add model label
    df_ft['model'] = 'finetuned'
    df_zs['model'] = 'zero-shot'
    df_fs['model'] = 'few-shot'

    # Keep only the examples that appear in all 3 (by input_context)
    shared_inputs = set(df_ft['input_context']) & set(df_zs['input_context']) & set(df_fs['input_context'])
    df_ft = df_ft[df_ft['input_context'].isin(shared_inputs)]
    df_zs = df_zs[df_zs['input_context'].isin(shared_inputs)]
    df_fs = df_fs[df_fs['input_context'].isin(shared_inputs)]

    # Optionally sort by input_context for consistency
    df_ft = df_ft.sort_values(by='input_context')
    df_zs = df_zs.sort_values(by='input_context')
    df_fs = df_fs.sort_values(by='input_context')

    # Combine them
    df_all = pd.concat([df_ft, df_zs, df_fs])

    # Melt the long-format DataFrame for plotting
    melted = df_all.melt(
        id_vars='model',
        value_vars=['bleu', 'rouge1', 'rougeL', 'meteor', 'bart_score'],
        var_name='metric',
        value_name='score'
    )

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x='metric', y='score', hue='model', palette='pastel')
    plt.title('Metric Comparison Across Models (Same 1000 Examples)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/eval/plots/metric_comparison_across_models.png')
    plt.show()

    # Print how many shared examples we used
    print(f"Used {len(shared_inputs)} shared examples across all models.")


def plot_bar_comparison():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    data = {
        'Metric': ['Perplexity', 'MAUVE', 'Self-BLEU'],
        'Fine-tuned': [92.6580, 0.0077, 0.5223],
        'Zero-shot': [2797.0630, 0.0509, 0.0786],
        'Few-shot': [52.9779, 0.0093, 0.4824]
    }

    df = pd.DataFrame(data)
    df_melt = df.melt(id_vars='Metric', var_name='Model', value_name='Value')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melt, x='Metric', y='Value', hue='Model', palette='pastel')
    plt.title('Model Comparison on Perplexity, MAUVE, and Self-BLEU')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/eval/plots/summary_barplot_perplexity_mauve_selfbleu.png")
    plt.show()

def plot_all_finetuned_metrics():
    plot_performance_distribution(df_finetuned, "Finetuned")
    plot_metric_correlation_heatmap(df_finetuned, "Finetuned")
    plot_error_analysis(df_finetuned, "Finetuned")

def plot_all_zero_shot_metrics():    
    # Plotting functions for zero-shot metrics
    plot_performance_distribution(df_zero_shot, "Zero-Shot")
    plot_metric_correlation_heatmap(df_zero_shot, "Zero-Shot")
    plot_error_analysis(df_zero_shot, "Zero-Shot")


def plot_all_few_shot_metrics():
    # Plotting functions for few-shot metrics
    plot_performance_distribution(df_few_shot, "Few-Shot")
    plot_metric_correlation_heatmap(df_few_shot, "Few-Shot")
    plot_error_analysis(df_few_shot, "Few-Shot")



if __name__ == "__main__":
    plot_all_finetuned_metrics()
    plot_all_zero_shot_metrics()
    plot_all_few_shot_metrics()
    plot_metrics_on_shared_subset()

