import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightning import pytorch as pl
from sklearn.model_selection import train_test_split
import torch
from chemprop import data, featurizers, models, nn

seed = 1

pl.seed_everything(seed)

input_path = r"C:\Users\rhys-\OneDrive\data_hnrs\spectra\nist\normalised_nist_exp_spectra.parquet"  # path to your data .csv file
num_workers = (
    0  # number of workers for dataloader. 0 means using main process for data loading
)
smiles_column = "smiles"  # name of the column containing SMILES strings
target_columns = np.arange(400, 4002, 2).astype(
    str
)  # list of names of the columns containing targets

df_input = pd.read_parquet(input_path)

# check df_input['smiles'] for nan
df_input = df_input.dropna(subset=[smiles_column])

# check df_input['smiles] for none
df_input = df_input[df_input[smiles_column] != "none"]
df_input = df_input[df_input[smiles_column] != "nan"]
df_input = df_input[df_input[smiles_column] != "None"]

smis = df_input.loc[:, smiles_column].values
ys = df_input.loc[:, target_columns].values

all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]

train_data, val_test_data = train_test_split(all_data, test_size=0.1, random_state=seed)
val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=seed)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

train_dset = data.MoleculeDataset(train_data, featurizer)
val_dset = data.MoleculeDataset(val_data, featurizer)
test_dset = data.MoleculeDataset(test_data, featurizer)

train_loader = data.MolGraphDataLoader(train_dset, num_workers=num_workers, seed=seed)
val_loader = data.MolGraphDataLoader(
    val_dset, num_workers=num_workers, seed=seed, shuffle=False
)
test_loader = data.MolGraphDataLoader(
    test_dset, num_workers=num_workers, seed=seed, shuffle=False
)

mp = nn.BondMessagePassing(depth=6, d_h=2200, dropout=0.05)
agg = nn.MeanAggregation()
ffn = nn.RegressionFFN(
    input_dim=2200,
    n_layers=6,
    hidden_dim=2200,
    dropout=0.05,
    # loc=scaler.mean_, # pass in the mean of the training targets
    # scale=scaler.scale_,
    n_tasks=1801,  # pass in the scale of the training targets
)

metric_list = [
    nn.metrics.SIDMetric()
]  # Only the first metric is used for training and early stopping


batch_norm = False
# metric_list = [nn.metrics.RMSEMetric()] # Only the first metric is used for training and early stopping
mpnn = models.MPNN(mp, agg, ffn, batch_norm)

trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=False,  # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=10,  # number of epochs to train for
)

trainer.fit(mpnn, train_loader, val_loader)

preds = trainer.predict(mpnn, test_loader)

print(preds[0])
