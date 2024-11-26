import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn

# Your existing Dataset class and get_dataset function should be here
# from your_dataset_module import Dataset, get_dataset
from source.utils import *
from source.modelsVAE import *

if os.name == "nt":
    loading_dir = "C:/Users/jed95/Documents/GitHub/CVAE/output/"
else:
    loading_dir = "/home/adlink3/Downloads/CVAE/output/"


# Define your sequence creation function
def create_sequences(values, labels, sequence_length):
    sequences = []
    seq_labels = []
    for i in range(len(values) - sequence_length + 1):
        seq = values[i : i + sequence_length]
        label = labels[
            i + sequence_length - 1
        ]  # Label of the last element in the sequence
        sequences.append(seq)
        seq_labels.append(label)
    return np.array(sequences), np.array(seq_labels)


# Load and process data function
def load_and_process_data(sequence_length):
    # Initialize lists to hold all sequences and labels
    all_sequences = []
    all_labels = []

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Load all CSV files in the A2Benchmark directory
    if os.name == "nt":
        data_dir = "C:/Users/jed95/Documents/GitHub/anomaly_detection/dataset/yahoo_s5/A2Benchmark/"
    else:
        data_dir = "/home/adlink3/Downloads/yahoo_s5/A2Benchmark/"
    data_files = glob.glob(data_dir + "*.csv")
    for file in data_files:
        df = pd.read_csv(file)
        # Ensure the DataFrame has the necessary columns
        if {"value", "is_anomaly"}.issubset(df.columns):
            # Reshape 'value' for the scaler
            values = df["value"].values.reshape(-1, 1)
            labels = df["is_anomaly"].values

            # Fit and transform the values using MinMaxScaler
            scaled_values = scaler.fit_transform(values).flatten()

            # Create sequences from the scaled values
            sequences, seq_labels = create_sequences(
                scaled_values, labels, sequence_length
            )
            all_sequences.append(sequences)
            all_labels.append(seq_labels)
        else:
            print(f"Skipping file {file} as it does not contain the required columns.")

    # Concatenate all sequences and labels
    all_sequences = np.concatenate(all_sequences, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_sequences, all_labels


# Set your desired sequence length
sequence_length = 10

# Load and process the data
sequences, labels = load_and_process_data(sequence_length)

# Reshape data to have shape (data_length, sequence_length, num_params)
num_params = 1  # Since the data is univariate
sequences = sequences.reshape(-1, sequence_length, num_params)

# Transpose data if necessary
sequences = np.transpose(
    sequences, axes=(0, 2, 1)
)  # Shape: (data_length, num_params, sequence_length)

# Output shapes
print("Shape of the sequences:", sequences.shape)
print("Shape of the labels:", labels.shape)

# Stratified K-Fold Cross-Validation
n_splits = 5  # Number of folds
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_indices, test_indices) in enumerate(skf.split(sequences, labels)):
    print(f"\nFold {fold + 1}/{n_splits}")
    # Select only normal data (label == 0) for training
    train_indices_normal = train_indices[labels[train_indices] == 0]
    x_train = sequences[train_indices_normal]
    y_train = labels[train_indices_normal]  # Should be all zeros

    # Test data includes both normal and anomalous data
    x_test = sequences[test_indices]
    y_test = labels[test_indices]

    # Output shapes
    print("Training data shape:", x_train.shape)
    print("Testing data shape:", x_test.shape)
    print("Testing labels distribution:", np.unique(y_test, return_counts=True))

    # Convert data to float32
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # Preparing training and testing datasets
    train_dataset = Dataset(x_train)
    test_dataset = Dataset(x_test)

    # Preparing the data loader
    batch_size = 64
    train_data, test_data = get_dataset(train_dataset, test_dataset, batch_size)

    # Define your model parameters
    num_epochs = 10  # Number of epochs for training
    latent_dim = 10  # default:32 dimension of the latent space
    beta = 0.001  # value of beta that controls the regularization
    metric = "MSE"  # Default:BCE choice of metric for calculating reconstruction error
    training = 1  # whether to train the model or load a trained one

    # Naming the model for saving
    model_name = f"Demo_VAE_s5_l_{latent_dim}_beta_{beta}_batch_{batch_size}_metric_{metric}_fold_{fold + 1}"

    # Initiating the model
    model = VAE(
        latent_dim=latent_dim,
        num_param=num_params,
        window_size=sequence_length,
        scale_flag=0,
    )

    # Setting up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training or loading the trained model
    if training:
        model = train_model(
            model,
            optimizer,
            model_name,
            train_data,
            test_data,
            loading_dir,
            metric,
            beta,
            num_epochs,
            save=True,
            verbose=1,
        )
    else:
        model.load_state_dict(
            torch.load(
                (loading_dir + model_name + ".pth"), map_location=torch.device("cpu")
            )
        )
        print("Model loaded!")

    # Here you can add evaluation metrics per fold if needed
    # For example, calculate reconstruction error on x_test and compare with y_test
