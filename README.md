# Decision Tree From Scratch

A decision tree classifier built entirely from scratch using only NumPy — no sklearn models, no ensemble methods. Trained and evaluated on the breast cancer dataset, achieving 93.85% accuracy and 97.18% recall. Implements entropy, information gain, best split search, recursive tree building, and manual evaluation metrics.

## Dataset
- **Breast Cancer Wisconsin Dataset** (via `sklearn.datasets`)
- 569 samples, 30 features
- Binary classification: Malignant vs Benign

## How It Works
1. **Entropy** — measures impurity at each node
2. **Information Gain** — determines how much a split reduces impurity
3. **Best Split** — searches every feature and threshold to find the optimal split
4. **Build Tree** — recursively grows the tree until max depth or pure nodes
5. **Predict** — traverses the tree for each sample
6. **Evaluate** — manually computes accuracy, precision, recall, and confusion matrix

## Results (max_depth=5)
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 93.85% |
| Recall    | 97.18% |
| Precision | 93.24% |

## Why Recall Matters
This is a cancer detection model. A false negative (malignant classified as benign) is far more dangerous than a false positive — so recall is the most critical metric here. The model achieves 97.18% recall, meaning it correctly catches 97% of actual malignant cases.

## Requirements
```
numpy
scikit-learn  # only used for loading the dataset and train/test split
```


