import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


def load_json_labels(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def create_label_df(json_data):
    return pd.DataFrame.from_dict(json_data, orient='index')


def plot_label_balance_stacked(df, title, save_path):
    """
    Plot the positive/negative balance for each label using stacked bars and save as PNG.
    """
    label_balance = df.mean()
    label_balance_negative = 1 - label_balance

    plt.figure(figsize=(14, 6))
    bar_width = 0.8

    labels = label_balance.index
    pos_bars = plt.bar(labels, label_balance, bar_width, label='Positive', color='#2ecc71')
    neg_bars = plt.bar(labels, label_balance_negative, bar_width, bottom=label_balance, label='Negative',
                       color='#e74c3c')

    plt.title(f'Label Balance - {title}')
    plt.xlabel('Labels')
    plt.ylabel('Proportion')
    plt.legend(title='Class')
    plt.xticks(rotation=45, ha='right')

    # Add percentage labels on the bars
    for i, (pos, neg) in enumerate(zip(label_balance, label_balance_negative)):
        plt.text(i, pos / 2, f'{pos:.1%}', ha='center', va='center', color='white', fontweight='bold')
        plt.text(i, pos + neg / 2, f'{neg:.1%}', ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_label_distribution_across_splits_stacked(train_df, val_df, test_df, save_path):
    """
    Plot the distribution of positive and negative labels across train, validation, and test splits and save as PNG.
    """
    train_dist = train_df.mean()
    val_dist = val_df.mean()
    test_dist = test_df.mean()

    df = pd.DataFrame({
        'Train Positive': train_dist,
        'Train Negative': 1 - train_dist,
        'Validation Positive': val_dist,
        'Validation Negative': 1 - val_dist,
        'Test Positive': test_dist,
        'Test Negative': 1 - test_dist
    })

    plt.figure(figsize=(16, 6))
    df.plot(kind='bar', stacked=True, width=0.8)
    plt.title('Label Distribution Across Splits')
    plt.xlabel('Labels')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Split and Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_sample_counts(train_df, val_df, test_df, save_path):
    """
    Plot the number of samples in each split and save as PNG.
    """
    counts = [len(train_df), len(val_df), len(test_df)]
    splits = ['Train', 'Validation', 'Test']

    plt.figure(figsize=(4, 6))
    bars = plt.bar(splits, counts)
    plt.title('Number of Samples in Each Split')
    plt.ylabel('Number of Samples')

    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:,}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# Get the directory of the JSON files
json_dir = os.path.dirname('/Volumes/SSD/BVIArtefact_8_crops_all_videos/train_labels.json')

# Load the data
train_data = load_json_labels(os.path.join(json_dir, 'train_labels.json'))
val_data = load_json_labels(os.path.join(json_dir, 'val_labels.json'))
test_data = load_json_labels(os.path.join(json_dir, 'test_labels.json'))

# Create DataFrames
train_df = create_label_df(train_data)
val_df = create_label_df(val_data)
test_df = create_label_df(test_data)

# Generate and save plots
plot_label_balance_stacked(train_df, 'Train Set', os.path.join(json_dir, 'label_balance_train.png'))
plot_label_balance_stacked(val_df, 'Validation Set', os.path.join(json_dir, 'label_balance_val.png'))
plot_label_balance_stacked(test_df, 'Test Set', os.path.join(json_dir, 'label_balance_test.png'))
plot_label_distribution_across_splits_stacked(train_df, val_df, test_df,
                                              os.path.join(json_dir, 'label_distribution_across_splits.png'))
plot_sample_counts(train_df, val_df, test_df, os.path.join(json_dir, 'sample_counts.png'))

print(f"Plots have been saved in the directory: {json_dir}")
