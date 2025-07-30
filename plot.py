import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the finetuned CSV
df_finetuned = pd.read_csv('finetuned_detailed_analysis.csv')
df_finetuned['model'] = 'finetuned'  # Tag for the model type

def plot_finetuned_performance_distribution():
    # Plot box plot of BARTScore
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df_finetuned, x='model', y='bart_score', palette='pastel')

    plt.title('Finetuned BARTScore Distribution')
    plt.ylabel('BARTScore')
    plt.xlabel('')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('finetuned_bartscore_boxplot.png')
    plt.show()

def plot_finetuned_metric_correlation_heatmap():
    # Compute the correlation matrix
    metric_cols = ['bleu', 'rouge1', 'rougeL', 'meteor', 'bart_score']
    df_metrics = df_finetuned[metric_cols]
    corr = df_metrics.corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Finetuned Metric Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('finetuned_metric_correlation_heatmap.png')
    plt.show()

def plot_finetuned_error_analysis():
    # Add input length column (in tokens or characters)
    df_finetuned['input_length'] = df_finetuned['input_context'].apply(lambda x: len(x.split()))  # token-based
    # OR: use len(x) for character length

    # Scatter plot: BARTScore vs input length
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_finetuned, x='input_length', y='bart_score', alpha=0.5)
    sns.regplot(data=df_finetuned, x='input_length', y='bart_score', scatter=False, color='red', ci=None)

    plt.title('BARTScore vs Input Length (finetuned)')
    plt.xlabel('Input Length (Tokens)')
    plt.ylabel('BARTScore')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('finetuned_bartscore_vs_length.png')
    plt.show()

def plot_metric_comparison_bar_plot():
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_finetuned, x='model', y='bart_score', palette='pastel')
    plt.title('Finetuned BARTScore Comparison')
    plt.ylabel('BARTScore')
    plt.xlabel('Model')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('finetuned_bartscore_comparison.png')
    plt.show()

