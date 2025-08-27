import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def extract_features(p1: str, p2: str):
    p1_vector = np.array(singles_df.loc[p1])
    p2_vector = np.array(singles_df.loc[p2])

    difference = abs(p1_vector - p2_vector)

    mean_p1 = np.mean(p1_vector)
    mean_p2 = np.mean(p2_vector)

    std_p1 = np.std(p1_vector)
    std_p2 = np.std(p2_vector)

    features = np.concatenate([
        p1_vector, 
        p2_vector, 
        difference, 
        np.array([mean_p1, mean_p2, std_p1, std_p2])
    ])

    return features

train_df = pd.read_csv('data/train_set.csv')
test_df = pd.read_csv('data/test_set.csv', header=None)

single_perturb_cols = [col for col in train_df.columns if '+ctrl' in col]

singles_df = train_df[[train_df.columns[0]] + single_perturb_cols].T
pairs_df = train_df.drop(columns=[train_df.columns[0]] + single_perturb_cols).T

X = []
y = []

for row in pairs_df.index:
    if 'ctrl' in row:
        continue
    p1, p2 = row.split('+')
    if '.' in p2:
        p2, _ = p2.split('.')
    p1 += '+ctrl'
    p2 += '+ctrl'
    pair_row = np.array(pairs_df.loc[row])
    features = extract_features(p1=p1, p2=p2)
    X.append(features)
    y.append(pair_row)

X_test = []

for pair in test_df.iloc[:, 0]:
    p1, p2 = pair.split('+')
    p1 += '+ctrl'
    p2 += '+ctrl'
    features = extract_features(p1=p1, p2=p2)
    X_test.append(features)

X = np.array(X)
y = np.array(y)
X_test = np.array(X_test)

x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)
X_test_scaled = x_scaler.transform(X_test)

y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500)
model.fit(X_scaled, y_scaled)

y_pred_scaled = model.predict(X_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

output_rows = []
gene_names = train_df.iloc[:, 0].tolist()

for i, pair in enumerate(test_df.iloc[:, 0]):
    for j, gene in enumerate(gene_names):
        output_rows.append({
            'gene': gene,
            'perturbation': pair,
            'expression': y_pred[i][j]
        })

predictions_df = pd.DataFrame(output_rows)
predictions_df.to_csv('prediction.csv', index=False)