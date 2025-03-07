import torch
import math
import time
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels  # Labels can be None for unlabeled data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # Assuming data shape is [num_param, window_size]
        X = self.data[index, :, :]
        if self.labels is not None:
            y = self.labels[index]
            return X, y
        else:
            return X


def get_dataset(train, validation, batch_size=128):
    """Creates dataloaders for training and validation datasets."""
    from torch.utils.data.sampler import SubsetRandomSampler

    def get_sampler(data):
        num_data = np.shape(data)[0]
        sampler = SubsetRandomSampler(torch.from_numpy(np.arange(0, num_data)))
        return sampler

    train_data = torch.utils.data.DataLoader(
        train, batch_size=batch_size, pin_memory=cuda, sampler=get_sampler(train.data)
    )
    valid_data = torch.utils.data.DataLoader(
        validation,
        batch_size=batch_size,
        pin_memory=cuda,
        sampler=get_sampler(validation.data),
    )

    return train_data, valid_data


def likelihood_loss(r, x, metric="BCE"):
    """
    Calculates the likelihood loss between the reconstruction r and input x.
    For our purposes with continuous data, we use the MSE loss.
    """
    r = r.view(r.size(0), -1)
    x = x.view(x.size(0), -1)
    if metric == "BCE":
        loss = -torch.sum(
            x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1
        )
    elif metric == "MSE":
        mse_loss = torch.nn.MSELoss(reduction="none")
        loss = torch.sum(mse_loss(x, r), dim=-1)
    return loss


def train_model(
    model,
    optimizer,
    train_loader,
    num_epochs=10,
    save=True,
    save_dir="./",
    model_name="model",
):
    """
    Training function for the supervised Transformer Autoencoder.
    The loss is the sum of:
      - Reconstruction loss: MSE between the input and its reconstruction.
      - Classification loss: Cross-entropy between predicted class logits and ground-truth labels.
    The KL divergence term is removed.
    """
    time_start = time.time()
    model = model.to(device)
    model.train()

    # Arrays to store loss values per epoch
    training_total_loss = np.zeros(num_epochs)
    training_rec_loss = np.zeros(num_epochs)
    training_class_loss = np.zeros(num_epochs)

    num_batches = len(train_loader)
    for epoch in range(num_epochs):
        total_loss = 0.0
        rec_loss_total = 0.0
        class_loss_total = 0.0

        for X_batch, y_batch in train_loader:
            # For the Transformer model, the input shape is assumed to be [batch, num_param, window_size]
            if X_batch.dim() == 2:
                X_batch = X_batch.unsqueeze(
                    1
                )  # Now shape becomes [batch, 1, window_size]
            X_batch = X_batch.to(
                device
            )  # Remove unsqueeze(1) since it's already the expected shape
            y_batch = y_batch.to(device)

            # Create one-hot labels if your model expects them (here it's optional, as our forward accepts y_onehot)
            y_onehot = F.one_hot(y_batch, num_classes=model.num_classes).float()

            optimizer.zero_grad()
            # Forward pass: get reconstruction and class logits from the model
            x_rec, class_logits = model(X_batch, y_onehot)

            # Compute the reconstruction loss (MSE)
            rec_loss = F.mse_loss(x_rec, X_batch)
            # Compute the classification loss (cross entropy)
            class_loss = F.cross_entropy(class_logits, y_batch)

            loss = rec_loss + class_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            rec_loss_total += rec_loss.item()
            class_loss_total += class_loss.item()

        avg_total_loss = total_loss / num_batches
        avg_rec_loss = rec_loss_total / num_batches
        avg_class_loss = class_loss_total / num_batches

        training_total_loss[epoch] = avg_total_loss
        training_rec_loss[epoch] = avg_rec_loss
        training_class_loss[epoch] = avg_class_loss

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {avg_total_loss:.4f}, "
            f"Rec Loss: {avg_rec_loss:.4f}, Class Loss: {avg_class_loss:.4f}"
        )

    if save:
        torch.save(model.state_dict(), save_dir + model_name + ".pth")
        np.savez_compressed(
            save_dir + model_name + "_training_loss.npz",
            training_total_loss=training_total_loss,
            training_rec_loss=training_rec_loss,
            training_class_loss=training_class_loss,
        )
    time_end = time.time()
    print("Time elapsed: ", time_end - time_start)
    return model


def train_transformer_supervised_model_full(
    model, optimizer, train_loader, num_epochs=10
):
    """
    An alternate training loop without loss logging.
    """
    model = model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        rec_loss_total = 0.0
        class_loss_total = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(
                device
            )  # Input already has the shape [batch, num_param, window_size]
            y_batch = y_batch.to(device)

            y_onehot = F.one_hot(y_batch, num_classes=model.num_classes).float()

            optimizer.zero_grad()
            x_rec, class_logits = model(X_batch, y_onehot)
            rec_loss = F.mse_loss(x_rec, X_batch)
            class_loss = F.cross_entropy(class_logits, y_batch)
            loss = rec_loss + class_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            rec_loss_total += rec_loss.item()
            class_loss_total += class_loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss:.4f}, "
            f"Rec Loss: {rec_loss_total:.4f}, Class Loss: {class_loss_total:.4f}"
        )
    return model


def find_transformer_score(model, data, metric="MSE", num_sample=50):
    """
    Compute an anomaly score based on reconstruction error and predicted class probability.
    The anomaly score is defined here as the mean reconstruction error multiplied by the probability
    of the anomaly class (assumed to be class index 1).
    """
    model = model.to("cpu")
    model.eval()
    num_data = np.shape(data)[0]
    data_tensor = torch.tensor(data).float()
    anomaly_score = np.zeros((num_data, num_sample))
    class_probs = np.zeros((num_data, model.num_classes))

    with torch.no_grad():
        for i in range(num_sample):
            x_rec, class_logits = model(data_tensor)
            if metric == "MSE":
                # Calculate the mean MSE per sample over all features
                loss = F.mse_loss(x_rec, data_tensor, reduction="none")
                loss = loss.view(loss.size(0), -1).mean(dim=1)
            else:
                loss = likelihood_loss(x_rec, data_tensor, metric)
            anomaly_score[:, i] = loss.numpy()
            class_probs += F.softmax(class_logits, dim=1).numpy()

    avg_anomaly_score = np.mean(anomaly_score, axis=1)
    avg_class_probs = class_probs / num_sample

    # Combine the reconstruction error and anomaly class probability (assumed to be class 1)
    anomaly_scores = avg_anomaly_score * avg_class_probs[:, 1]
    return anomaly_scores
