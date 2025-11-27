import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_EPOCH_LINE_RE = re.compile(
    r"Epoch\s*\[(\d+)/(\d+)\],\s*"
    r"Train Loss:\s*([0-9.]+),\s*"
    r"Val Loss:\s*([0-9.]+),\s*"
    r"Val Accuracy:\s*([0-9.]+),\s*"
    r"LR:\s*([0-9.]+),\s*"
    r"Time:\s*([0-9.]+)s"
)

_BEST_LINE_RE = re.compile(
    r"->\s*New best validation accuracy:\s*([0-9.]+)"
)

def parse_training_log(
    log_text: str,
) -> Tuple[List[Dict[str, float]], Optional[float]]:
    """
    Parse one or more lines of training log text into structured metrics.

    The function expects lines like:
        "Epoch [1/30], Train Loss: 0.7264, Val Loss: 0.5025, "
        "Val Accuracy: 0.8241, LR: 0.001000, Time: 5.47s"
        "  -> New best validation accuracy: 0.8241"

    Parameters
    ----------
    log_text:
        A single line or multi-line string containing training logs.

    Returns
    -------
    epochs:
        A list of dicts with keys:
        "epoch", "n_epochs", "train_loss", "val_loss",
        "val_accuracy", "lr", "time".
    best_val_acc:
        The best validation accuracy seen in the log (if any),
        otherwise ``None``.
    """
    epochs: List[Dict[str, float]] = []
    best_val_acc: Optional[float] = None

    for raw_line in log_text.splitlines():
        line = raw_line.strip()

        # Parse epoch metrics
        epoch_match = _EPOCH_LINE_RE.search(line)
        if epoch_match:
            (
                epoch_idx,
                n_epochs,
                train_loss,
                val_loss,
                val_acc,
                lr,
                time_s,
            ) = epoch_match.groups()

            epochs.append(
                {
                    "epoch": float(epoch_idx),
                    "n_epochs": float(n_epochs),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_accuracy": float(val_acc),
                    "lr": float(lr),
                    "time": float(time_s),
                }
            )

        # Parse "new best" lines (may be on separate lines)
        best_match = _BEST_LINE_RE.search(line)
        if best_match:
            best_val_acc = float(best_match.group(1))

    return epochs, best_val_acc


def save_training_curves_from_parsed(
    parsed_log: Tuple[List[Dict[str, float]], Optional[float]],
    output_dir: str,
    prefix: str
) -> None:
    """
    Given the output of :func:`parse_training_log`, create and save
    three line plots:

    1. Validation accuracy vs epoch
    2. Validation loss vs epoch
    3. Epoch index vs cumulative time up to that epoch

    Parameters
    ----------
    parsed_log:
        The tuple ``(epochs, best_val_acc)`` returned by ``parse_training_log``.
    output_dir:
        Directory where plots will be saved (created if it does not exist).
    prefix:
        Filename prefix for the generated plot files.
    """
    epochs, _ = parsed_log
    if not epochs:
        return

    os.makedirs(output_dir, exist_ok=True)

    epoch_idx = np.array([e["epoch"] for e in epochs], dtype=float)
    val_acc = np.array([e["val_accuracy"] for e in epochs], dtype=float)
    val_loss = np.array([e["val_loss"] for e in epochs], dtype=float)
    times = np.array([e["time"] for e in epochs], dtype=float)
    cum_time = np.cumsum(times)

    # 1) Validation accuracy vs epoch
    def _plot_metric(x, y, xlabel, ylabel, title, save_path, color=None):
        plt.figure(figsize=(6, 4))
        if color is not None:
            plt.plot(x, y, marker="o", color=color)
        else:
            plt.plot(x, y, marker="o")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    _plot_metric(
        epoch_idx,
        val_acc,
        xlabel="Epoch",
        ylabel="Validation Accuracy",
        title="Validation Accuracy vs Epoch",
        save_path=os.path.join(output_dir, f"{prefix}_val_accuracy_vs_epoch.png"),
    )
    _plot_metric(
        epoch_idx,
        val_loss,
        xlabel="Epoch",
        ylabel="Validation Loss",
        title="Validation Loss vs Epoch",
        save_path=os.path.join(output_dir, f"{prefix}_val_loss_vs_epoch.png"),
        color="tab:orange",
    )
    _plot_metric(
        epoch_idx,
        cum_time,
        xlabel="Epoch",
        ylabel="Cumulative Time (s)",
        title="Epoch vs Cumulative Training Time",
        save_path=os.path.join(output_dir, f"{prefix}_epoch_vs_cumulative_time.png"),
        color="tab:green",
    )




save_training_curves_from_parsed(parse_training_log("""
  Training final model with best hyperparameters for 50 epochs...
Using device: mps
Epoch [1/50], Train Loss: 0.6504, Val Loss: 0.4505, Val Accuracy: 0.8333, LR: 0.001000, Time: 10.22s
  -> New best validation accuracy: 0.8333
Epoch [2/50], Train Loss: 0.4123, Val Loss: 0.3554, Val Accuracy: 0.8742, LR: 0.001000, Time: 5.49s
  -> New best validation accuracy: 0.8742
Epoch [3/50], Train Loss: 0.3591, Val Loss: 0.3359, Val Accuracy: 0.8804, LR: 0.001000, Time: 5.82s
  -> New best validation accuracy: 0.8804
Epoch [4/50], Train Loss: 0.3270, Val Loss: 0.3594, Val Accuracy: 0.8662, LR: 0.001000, Time: 5.48s
Epoch [5/50], Train Loss: 0.3049, Val Loss: 0.3379, Val Accuracy: 0.8809, LR: 0.001000, Time: 5.34s
  -> New best validation accuracy: 0.8809
Epoch [6/50], Train Loss: 0.2854, Val Loss: 0.3199, Val Accuracy: 0.8801, LR: 0.001000, Time: 5.37s
Epoch [7/50], Train Loss: 0.2723, Val Loss: 0.2607, Val Accuracy: 0.9052, LR: 0.001000, Time: 5.60s
  -> New best validation accuracy: 0.9052
Epoch [8/50], Train Loss: 0.2567, Val Loss: 0.3169, Val Accuracy: 0.8853, LR: 0.001000, Time: 5.36s
Epoch [9/50], Train Loss: 0.2518, Val Loss: 0.2440, Val Accuracy: 0.9125, LR: 0.001000, Time: 5.37s
  -> New best validation accuracy: 0.9125
Epoch [10/50], Train Loss: 0.2356, Val Loss: 0.2741, Val Accuracy: 0.9033, LR: 0.001000, Time: 5.36s
Epoch [11/50], Train Loss: 0.2292, Val Loss: 0.2625, Val Accuracy: 0.9053, LR: 0.001000, Time: 5.38s
Epoch [12/50], Train Loss: 0.2190, Val Loss: 0.2868, Val Accuracy: 0.8969, LR: 0.001000, Time: 5.40s
Epoch [13/50], Train Loss: 0.2143, Val Loss: 0.2353, Val Accuracy: 0.9135, LR: 0.001000, Time: 5.34s
  -> New best validation accuracy: 0.9135
Epoch [14/50], Train Loss: 0.2092, Val Loss: 0.2322, Val Accuracy: 0.9143, LR: 0.001000, Time: 5.89s
  -> New best validation accuracy: 0.9143
Epoch [15/50], Train Loss: 0.2041, Val Loss: 0.2420, Val Accuracy: 0.9111, LR: 0.001000, Time: 5.64s
Epoch [16/50], Train Loss: 0.1973, Val Loss: 0.2359, Val Accuracy: 0.9162, LR: 0.001000, Time: 5.39s
  -> New best validation accuracy: 0.9162
Epoch [17/50], Train Loss: 0.1896, Val Loss: 0.2554, Val Accuracy: 0.9074, LR: 0.001000, Time: 5.39s
Epoch [18/50], Train Loss: 0.1871, Val Loss: 0.2478, Val Accuracy: 0.9112, LR: 0.001000, Time: 5.85s
Epoch [19/50], Train Loss: 0.1779, Val Loss: 0.2237, Val Accuracy: 0.9163, LR: 0.001000, Time: 5.50s
  -> New best validation accuracy: 0.9163
Epoch [20/50], Train Loss: 0.1749, Val Loss: 0.2407, Val Accuracy: 0.9114, LR: 0.001000, Time: 5.39s
Epoch [21/50], Train Loss: 0.1672, Val Loss: 0.2352, Val Accuracy: 0.9176, LR: 0.001000, Time: 5.34s
  -> New best validation accuracy: 0.9176
Epoch [22/50], Train Loss: 0.1655, Val Loss: 0.2400, Val Accuracy: 0.9116, LR: 0.001000, Time: 5.38s
Epoch [23/50], Train Loss: 0.1608, Val Loss: 0.2590, Val Accuracy: 0.9081, LR: 0.001000, Time: 5.30s
Epoch [24/50], Train Loss: 0.1559, Val Loss: 0.2267, Val Accuracy: 0.9177, LR: 0.001000, Time: 5.41s
  -> New best validation accuracy: 0.9177
Epoch [25/50], Train Loss: 0.1522, Val Loss: 0.2324, Val Accuracy: 0.9197, LR: 0.000500, Time: 5.53s
  -> New best validation accuracy: 0.9197
Epoch [26/50], Train Loss: 0.1268, Val Loss: 0.2060, Val Accuracy: 0.9245, LR: 0.000500, Time: 5.39s
  -> New best validation accuracy: 0.9245
Epoch [27/50], Train Loss: 0.1205, Val Loss: 0.2156, Val Accuracy: 0.9241, LR: 0.000500, Time: 5.34s
Epoch [28/50], Train Loss: 0.1181, Val Loss: 0.2164, Val Accuracy: 0.9209, LR: 0.000500, Time: 5.35s
Epoch [29/50], Train Loss: 0.1157, Val Loss: 0.2307, Val Accuracy: 0.9195, LR: 0.000500, Time: 5.36s
Epoch [30/50], Train Loss: 0.1143, Val Loss: 0.2294, Val Accuracy: 0.9167, LR: 0.000500, Time: 5.42s
Epoch [31/50], Train Loss: 0.1109, Val Loss: 0.2181, Val Accuracy: 0.9237, LR: 0.000500, Time: 5.36s
Epoch [32/50], Train Loss: 0.1065, Val Loss: 0.2207, Val Accuracy: 0.9215, LR: 0.000250, Time: 5.43s
Epoch [33/50], Train Loss: 0.0927, Val Loss: 0.2235, Val Accuracy: 0.9203, LR: 0.000250, Time: 5.54s
Epoch [34/50], Train Loss: 0.0898, Val Loss: 0.2078, Val Accuracy: 0.9264, LR: 0.000250, Time: 5.36s
  -> New best validation accuracy: 0.9264
Epoch [35/50], Train Loss: 0.0879, Val Loss: 0.2202, Val Accuracy: 0.9257, LR: 0.000250, Time: 5.36s
Epoch [36/50], Train Loss: 0.0858, Val Loss: 0.2106, Val Accuracy: 0.9273, LR: 0.000250, Time: 6.53s
  -> New best validation accuracy: 0.9273
Epoch [37/50], Train Loss: 0.0861, Val Loss: 0.2148, Val Accuracy: 0.9238, LR: 0.000250, Time: 5.37s
Epoch [38/50], Train Loss: 0.0834, Val Loss: 0.2263, Val Accuracy: 0.9219, LR: 0.000125, Time: 5.36s
Epoch [39/50], Train Loss: 0.0772, Val Loss: 0.2108, Val Accuracy: 0.9273, LR: 0.000125, Time: 5.33s
Epoch [40/50], Train Loss: 0.0759, Val Loss: 0.2151, Val Accuracy: 0.9268, LR: 0.000125, Time: 5.35s
Epoch [41/50], Train Loss: 0.0742, Val Loss: 0.2127, Val Accuracy: 0.9270, LR: 0.000125, Time: 5.38s
Epoch [42/50], Train Loss: 0.0741, Val Loss: 0.2148, Val Accuracy: 0.9265, LR: 0.000125, Time: 5.41s
Epoch [43/50], Train Loss: 0.0714, Val Loss: 0.2137, Val Accuracy: 0.9273, LR: 0.000125, Time: 5.33s
Epoch [44/50], Train Loss: 0.0711, Val Loss: 0.2119, Val Accuracy: 0.9265, LR: 0.000063, Time: 5.37s
Epoch [45/50], Train Loss: 0.0686, Val Loss: 0.2105, Val Accuracy: 0.9273, LR: 0.000063, Time: 5.32s
Epoch [46/50], Train Loss: 0.0670, Val Loss: 0.2124, Val Accuracy: 0.9267, LR: 0.000063, Time: 5.38s
Epoch [47/50], Train Loss: 0.0667, Val Loss: 0.2131, Val Accuracy: 0.9283, LR: 0.000063, Time: 5.66s
  -> New best validation accuracy: 0.9283
Epoch [48/50], Train Loss: 0.0666, Val Loss: 0.2129, Val Accuracy: 0.9280, LR: 0.000063, Time: 5.40s
Epoch [49/50], Train Loss: 0.0656, Val Loss: 0.2126, Val Accuracy: 0.9275, LR: 0.000063, Time: 5.31s
Epoch [50/50], Train Loss: 0.0648, Val Loss: 0.2150, Val Accuracy: 0.9268, LR: 0.000031, Time: 5.38s
"""), "submission/plots", "best_model")