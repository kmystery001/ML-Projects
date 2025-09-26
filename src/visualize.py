import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(metrics_list):
    df_metrics = pd.DataFrame(metrics_list).set_index('Model')
    df_metrics.plot(kind='bar', figsize=(12,6), rot=45)
    plt.title("Comparison of Classifier Metrics")
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()