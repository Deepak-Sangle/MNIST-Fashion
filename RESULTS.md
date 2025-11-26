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
