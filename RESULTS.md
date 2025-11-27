## Accuracy, loss, etc. when I used hyperparameters tunning without any k-fold

- When i used these hyperparams = {'batch_size': 64, 'learning_rate': 0.001, 'weight_decay': 1e-4, 'n_epochs': 30}, with RandomHorizontalFlip and RandomRotation transforms, then finally trained on with 50 epoches, without any k-fold validation, i got the final accuracy as this

```
Using device: cpu
Epoch [1/1], Train Loss: 3.0643, Val Loss: 2.3252, Val Accuracy: 0.0000
STUDENT_ID: 14263708
ACCURACY: 0.925900
PARAMETERS: 94410
TRAINING_CHECK: PASSED
```

- When I used 5 fold cross validation technique using 80 20 split, the best hyperparameter were {'batch_size': 64, 'learning_rate': 0.001, 'weight_decay': 0.0001, 'n_epochs': 8}. Training it for 50 epoch gives Epoch [50/50], Train Loss: 0.1633, Val Loss: 0.2079, Val Accuracy: 0.9254, LR: 0.000500, Time: 11.17s. 

```
K-fold summary over 5 folds: mean val accuracy = 0.0333, std = 0.0667, best = 0.1667
STUDENT_ID: 14263708
ACCURACY: 0.923700
PARAMETERS: 94410
TRAINING_CHECK: PASSED
```

- When i used a very basic model with only ToTensor() and Normalize() transforms, no k-fold technique and 30 epoch for 3 hypertuning, i get

```
--- Hyperparameter Set 1/3 ---
Batch size: 128, Learning rate: 0.001, Weight decay: 0.0001, Epochs: 30
Epoch [1/30], Train Loss: 0.6597, Val Loss: 0.4272, Val Accuracy: 0.8436, LR: 0.001000, Time: 5.84s
  -> New best validation accuracy: 0.8436
Epoch [2/30], Train Loss: 0.4122, Val Loss: 0.3582, Val Accuracy: 0.8762, LR: 0.001000, Time: 5.35s
  -> New best validation accuracy: 0.8762
...
Epoch [29/30], Train Loss: 0.1121, Val Loss: 0.2205, Val Accuracy: 0.9210, LR: 0.000500, Time: 5.32s
Epoch [30/30], Train Loss: 0.1100, Val Loss: 0.2192, Val Accuracy: 0.9228, LR: 0.000500, Time: 5.46s
Validation Accuracy for this hyperparameter set: 0.9253

--- Hyperparameter Set 2/3 ---
Batch size: 256, Learning rate: 0.001, Weight decay: 0.001, Epochs: 30
Epoch [1/30], Train Loss: 0.7264, Val Loss: 0.5025, Val Accuracy: 0.8241, LR: 0.001000, Time: 5.47s
...
Epoch [30/30], Train Loss: 0.1571, Val Loss: 0.2207, Val Accuracy: 0.9217, LR: 0.000250, Time: 9.29s

--- Hyperparameter Set 3/3 ---
Batch size: 512, Learning rate: 0.005, Weight decay: 0.0001, Epochs: 30
Epoch [1/30], Train Loss: 0.7457, Val Loss: 0.4949, Val Accuracy: 0.8257, LR: 0.005000, Time: 5.93s
  -> New best validation accuracy: 0.8257
...
Epoch [30/30], Train Loss: 0.1405, Val Loss: 0.2525, Val Accuracy: 0.9148, LR: 0.002500, Time: 9.00s

Validation Accuracy for this hyperparameter set: 0.9232
Best hyperparameters: {'batch_size': 128, 'learning_rate': 0.001, 'weight_decay': 0.0001, 'n_epochs': 30}
Best validation accuracy: 0.9253

Training final model with best hyperparameters for 50 epochs...
Epoch [1/50], Train Loss: 0.6504, Val Loss: 0.4505, Val Accuracy: 0.8333, LR: 0.001000, Time: 10.22s
  -> New best validation accuracy: 0.8333
Epoch [2/50], Train Loss: 0.4123, Val Loss: 0.3554, Val Accuracy: 0.8742, LR: 0.001000, Time: 5.49s
  -> New best validation accuracy: 0.8742
Epoch [49/50], Train Loss: 0.0656, Val Loss: 0.2126, Val Accuracy: 0.9275, LR: 0.000063, Time: 5.31s
Epoch [50/50], Train Loss: 0.0648, Val Loss: 0.2150, Val Accuracy: 0.9268, LR: 0.000031, Time: 5.38s
```

```
Using device: cpu
Epoch [1/1], Train Loss: 2.8747, Val Loss: 2.2810, Val Accuracy: 0.1667, LR: 0.001000, Time: 0.02s
  -> New best validation accuracy: 0.1667
STUDENT_ID: 14263708
ACCURACY: 0.922800
PARAMETERS: 94410
TRAINING_CHECK: PASSED
```

