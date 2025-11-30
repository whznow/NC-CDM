# NeuralCDM with Simplex Equiangular Regularization

## Project Overview

The core goal of this project is to predict whether a student will answer an exercise correctly based on:

1. Student knowledge state (modeled via embeddings)
2. Exercise characteristics (difficulty, discrimination, and knowledge relevance)
3. Regularization of embeddings to form meaningful clusters (using simplex equiangular loss and within-class loss)

The model combines a neural network for cognitive diagnosis with clustering-based regularization to enhance the interpretability and generalization of embeddings.

## File Structure

```
├── data/
│   └── ASSIST/                # Dataset directory
│       ├── log_data.json      # Raw log data (input)
│       ├── train_set.json     # Processed training data (output)
│       ├── val_set.json       # Processed validation data (output)
│       └── test_set.json      # Processed test data (output)
├── model/
│   └── ASSIST/             # Model checkpoint directory (auto-created)
├── result/
│   └── model_val.txt          # Validation results log (auto-generated)
├── config.txt                 # Configuration file (student/exer/knowledge counts)
├── divide_data.py             # Script to split raw data into train/val/test sets
├── data_loader.py             # Data loaders for training/validation/testing
├── model.py                   # Core NeuralCDM model definition
├── regularizer.py             # Regularization loss functions (simplex + within-class)
├── clustering.py              # K-Means clustering for embedding labeling
├── train.py                   # Main training and validation script
└── README.md                  # Project documentation
```

## Environment Requirements

- Python 3.7+
- PyTorch 1.8+
- Scikit-learn
- NumPy
- Pandas
- JSON (built-in)

Install dependencies via pip:

```
pip install torch scikit-learn numpy pandas
```

## Data Preparation

### Input Data Format

The raw data (`log_data.json`) should follow this structure (per student):

```
[
  {
    "user_id": 1,
    "log_num": 10,  # Number of exercise responses
    "logs": [
      {
        "exer_id": 101,
        "score": 1,   # 1 = correct, 0 = incorrect
        "knowledge_code": [5, 8]  # Knowledge concepts tested by the exercise
      },
      ...
    ]
  },
  ...
]
```

### Data Processing

Run `divide_data.py` to split the raw data into training (60%), validation (20%), and test (20%) sets:

```
python divide_data.py
```

- Automatically filters out students with insufficient logs (note: the filtering threshold can be adjusted in the script).
- Shuffles data to ensure randomness.
- Generates `train_set.json`, `val_set.json`, and `test_set.json` in the `data/ASSIST/` directory.

## Training Instructions

### Configuration Setup

Edit `config.txt` to set the number of students, exercises, and knowledge concepts (matches your dataset):

```
# Number of Students, Number of Exercises, Number of Knowledge Concepts
6866,17746,123  # Example values (update with your data)
```

### Start Training

Run `train.py` with the specified device (CPU/CUDA) and number of epochs:

```
# Usage: python train.py {device} {epoch}
python train.py cuda:0 20  # Train on GPU (cuda:0) for 20 epochs
python train.py cpu 10     # Train on CPU for 10 epochs
```

### Training Details

- **Batch Size**: Fixed at 128 (configurable in `data_loader.py`).
- **Optimizer**: Adam with a learning rate of 0.001.
- **Regularization**: Applied from epoch 4 onwards (configurable in `train.py`).
- **Checkpoints**: Saved to `model/junyi_712/` after each epoch.
- **Validation**: Runs after each training epoch; results logged to `result/model_val.txt`.

## Key Components

### Model Architecture (model.py)

Implements the NeuralCDM model with three core components:

1. **Embedding Layers**:
   - `student_emb`: Student knowledge state embeddings (dimension = number of knowledge concepts).
   - `k_difficulty`: Exercise difficulty embeddings (dimension = number of knowledge concepts).
   - `e_discrimination`: Exercise discrimination scalar (scaled to [0, 10]).
2. **Prediction Network**:
   - Three fully connected layers with dropout (512 → 256 → 1).
   - Sigmoid activation for output (predicts correct answer probability).
   - Non-negative weight clipping to ensure meaningful parameter interpretations.

### Regularization Mechanisms (regularizer.py)

Two key regularization losses to improve embedding quality:

1. **Within-Class Loss (Lw)**:
   - Minimizes the sum of squared distances from samples to their cluster center (encourages compact clusters).
   - Used for both student and exercise embeddings.
2. **Simplex Equiangular Loss (Lb)**:
   - Encourages cluster centroids to form a simplex structure (pairwise cosine similarity ≈ -1/(K-1) for K clusters).
   - Ensures uniform separation between clusters (improves embedding discriminability).

### Clustering Integration (clustering.py + train.py)

- **K-Means Clustering**: Runs on student/exercise embeddings every epoch (after the start regularization epoch).
- **Cluster Labels**: Used to compute within-class (Lw) and between-class (Lb) regularization losses.
- **Differentiable Loss**: Lb is computed on centroids derived from cluster labels, maintaining end-to-end training.

## Evaluation Metrics

After each training epoch, the model is validated on the validation set with three metrics:

1. **Accuracy**: Proportion of correct predictions (threshold = 0.5).
2. **RMSE**: Root Mean Squared Error between predicted probabilities and true scores.
3. **AUC**: Area Under the ROC Curve (measures ranking ability).

Results are printed to the console and logged to `result/model_val.txt`.

## Hyperparameters

Key hyperparameters (configurable in `train.py`):

| Parameter               | Description                                 | Default Value |
| ----------------------- | ------------------------------------------- | ------------- |
| K_student               | Number of student clusters                  | 39            |
| K_exer                  | Number of exercise clusters                 | 37            |
| student_reg_lambda_w    | Weight for student within-class loss (Lw)   | 0.0095        |
| student_reg_lambda_b    | Weight for student between-class loss (Lb)  | 0.0799        |
| exer_reg_lambda_w       | Weight for exercise within-class loss (Lw)  | 0.6760        |
| exer_reg_lambda_b       | Weight for exercise between-class loss (Lb) | 0.1546        |
| start_student_reg_epoch | Epoch to start student regularization       | 4             |
| start_exer_reg_epoch    | Epoch to start exercise regularization      | 4             |
| lr                      | Learning rate (Adam optimizer)              | 0.001         |
| batch_size              | raining batch size (in `data_loader.py`)    | 128           |

Adjust these hyperparameters based on your dataset size and characteristics for optimal performance.

## Notes

- **Device Compatibility**: The script automatically detects CUDA if available; specify the device explicitly when running `train.py`.
- **Empty Clusters**: Handled by adding small noise to centroids (avoids runtime errors).
- **Model Checkpoints**: Saved after each epoch for resuming training or inference.

For questions or issues, please refer to the code comments or contact the project maintainer.