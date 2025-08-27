## Data Explanation

This project was developed as part of the 2025 GIDS Biomedical Data Science Hackathon. 
The goal was to predict gene expression when two genes are over-expressed.
The dataset comes from experiments where different genes are perturbed in cells, and the resulting gene expression is measured.
Each cell contains only a subset of genes, making the data sparse.
Predicting how pairs of genes affect expression helps us understand complex genetic interactions relevant to human disease.

## Type of Model

We used a Multi-Layer Perceptron (MLP) Regressor to predict gene expression values. Key parameters:

Hidden layers: 2 layers with 128 and 64 neurons, respectively

Activation function: default ReLU

Maximum iterations: 500

Features and targets were standardized using StandardScaler before training.

Feature vectors were constructed by combining the expression values of single-gene perturbations, their absolute difference, and summary statistics (mean and standard deviation) of each gene.

## Performance

The model achieved:

RMSD: 0.16326 (on gene expression values ranging 0–5)

This demonstrates that the model predicts gene expression with high accuracy.

## Tech Stack

Language: Python

Libraries: NumPy, pandas, scikit-learn

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the model
python model.py
```

Once the model is finished running its output will be located in the main directory within the file `prediction.csv`