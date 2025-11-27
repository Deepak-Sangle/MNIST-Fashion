import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _save_kfold_plots(
    fold_epoch_histories,
    fold_accuracies,
) -> None:
    """
    Helper to create and save K-fold training plots of validation accuracy vs epoch per fold (line plots)
    """
    if not fold_epoch_histories and not fold_accuracies:
        return

    plots_dir = os.path.join("submission", "plots", "1")
    os.makedirs(plots_dir, exist_ok=True)

    # 1) Plot validation accuracy vs epoch for each fold
    for fold_idx, history in fold_epoch_histories:
        if not history:
            continue
        plt.figure(figsize=(6, 4))
        epochs = range(1, len(history) + 1)
        plt.plot(epochs, history, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title(f"Validation Accuracy vs Epoch (Fold {fold_idx})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f"val_accuracy_fold_{fold_idx}.png")
        plt.savefig(plot_path)
        plt.close()

def _save_hyperparam_config_plot(config_labels, config_val_accuracies) -> None:
    """
    Helper to create and save a plot of validation accuracy per hyperparameter configuration
    """
    if not config_labels or not config_val_accuracies:
        return

    plots_dir = os.path.join("submission", "plots", "1")
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(7, 4))
    x = np.arange(len(config_labels))
    plt.bar(x, config_val_accuracies)
    plt.xticks(x, config_labels, rotation=20, ha="right")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy per Hyperparameter Configuration")
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "val_accuracy_per_hyperparam_config.png")
    plt.savefig(plot_path)
    plt.close()
