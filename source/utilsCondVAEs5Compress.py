import torch
import math
import time
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


class Dataset(torch.utils.data.Dataset):  # deprecated
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels  # Labels can be None for unlabeled data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        X = self.data[index, :, :]
        if self.labels is not None:
            y = self.labels[index]
            return X, y
        else:
            return X


def get_dataset(train, validation, batch_size=128):  # deprecated
    """this function creates the dataloader for the training and validation
    data to be used for training.

    Args:
        train (torch dataset): trianing dataset
        validation (torch dataset): validation dataset
        batch_size (int, optional): batch size for training. Defaults to 128.

    Returns:
        dataloaders: training and validation dataloaders
    """

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


def likelihood_loss(
    r, x, metric="BCE"
):  # ChatGPT(#CondVAE): Since the data might not be binary, consider using Mean Squared Error (MSE) loss.
    """calculates likelihood loss between input and its reconstruction

    Args:
        r (tensor): reconstructed data
        x (tensor): input data

    Returns:
        tensor: likelihood loss between reconstructed and input data
    """
    r = r.view(r.size()[0], -1)
    x = x.view(x.size()[0], -1)
    likelihood_loss = -torch.sum(
        x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1
    )

    if metric == "BCE":
        likelihood_loss = -torch.sum(
            x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1
        )
    elif metric == "MSE":
        mse_loss = torch.nn.MSELoss(reduction="none")
        likelihood_loss = torch.sum(mse_loss(x, r), dim=-1)

    return likelihood_loss


def train_model(
    model,
    optimizer,
    train_loader,
    num_epochs=10,
    save=True,
    save_dir="./",
    model_name="model",
):
    time_start = time.time()
    model = model.to(device)
    model.train()

    # Initialize arrays to store losses per epoch
    training_total_loss = np.zeros(num_epochs)
    training_rec_loss = np.zeros(num_epochs)
    training_kl_loss = np.zeros(num_epochs)
    training_class_loss = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        total_loss = 0
        rec_loss_total = 0
        kl_loss_total = 0
        class_loss_total = 0
        num_batches = len(train_loader)

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.unsqueeze(1).to(
                device
            )  # Shape: [batch_size, num_param, window_size]
            y_batch = y_batch.to(device)

            y_onehot = F.one_hot(y_batch, num_classes=model.num_classes).float()

            optimizer.zero_grad()
            z, x_rec, class_logits = model(X_batch, y_onehot)

            # Reconstruction loss
            likelihood = -likelihood_loss(x_rec, X_batch, metric="MSE")
            rec_loss = torch.mean(-likelihood)

            # KL divergence
            kl_div = torch.mean(model.kl_div)

            # Classification loss
            class_loss = F.cross_entropy(class_logits, y_batch)

            # Total loss
            loss = rec_loss + kl_div + class_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            rec_loss_total += rec_loss.item()
            kl_loss_total += kl_div.item()
            class_loss_total += class_loss.item()

        num_batches = 1  # TODO: delete this
        # print(num_batches)
        # Compute average losses per epoch
        avg_total_loss = total_loss / num_batches
        avg_rec_loss = rec_loss_total / num_batches
        avg_kl_loss = kl_loss_total / num_batches
        avg_class_loss = class_loss_total / num_batches

        # Store losses in arrays
        training_total_loss[epoch] = avg_total_loss
        training_rec_loss[epoch] = avg_rec_loss
        training_kl_loss[epoch] = avg_kl_loss
        training_class_loss[epoch] = avg_class_loss

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Total Loss: {avg_total_loss:.4f}, "
            f"Rec Loss: {avg_rec_loss:.4f}, "
            f"KL Div: {avg_kl_loss:.4f}, "
            f"Class Loss: {avg_class_loss:.4f}"
        )

    if save:
        torch.save(model.state_dict(), (save_dir + model_name + ".pth"))
        np.savez_compressed(
            (save_dir + model_name + "_training_loss"),
            training_total_loss=training_total_loss,
            training_rec_loss=training_rec_loss,
            training_kl_loss=training_kl_loss,
            training_class_loss=training_class_loss,
        )
    time_end = time.time()
    print("Time elapsed: ", time_end - time_start)
    return model


def train_model_full(model, optimizer, train_loader, num_epochs=10):  # deprecated
    model = model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        rec_loss_total = 0
        kl_loss_total = 0
        class_loss_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.unsqueeze(1).to(
                device
            )  # Shape: [batch_size, num_param, window_size]
            y_batch = y_batch.to(device)

            y_onehot = F.one_hot(y_batch, num_classes=model.num_classes).float()

            optimizer.zero_grad()
            x_rec, class_logits = model(X_batch, y_onehot)

            # Reconstruction loss
            likelihood = -likelihood_loss(x_rec, X_batch, metric="MSE")
            rec_loss = torch.mean(-likelihood)

            # KL divergence
            kl_div = torch.mean(model.kl_div)

            # Classification loss
            class_loss = F.cross_entropy(class_logits, y_batch)

            # Total loss
            loss = rec_loss + kl_div + class_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            rec_loss_total += rec_loss.item()
            kl_loss_total += kl_div.item()
            class_loss_total += class_loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Total Loss: {total_loss:.4f}, "
            f"Rec Loss: {rec_loss_total:.4f}, "
            f"KL Div: {kl_loss_total:.4f}, "
            f"Class Loss: {class_loss_total:.4f}"
        )
    return model


def find_score(model, data, metric="MSE", num_sample=50):  # deprecated
    model = model.to("cpu")
    model.eval()
    num_data = np.shape(data)[0]
    data_tensor = torch.tensor(data).float()
    anomaly_score = np.zeros((num_data, num_sample))
    class_probs = np.zeros((num_data, model.num_classes))

    with torch.no_grad():
        for i in range(num_sample):
            x_rec, class_logits = model(data_tensor)
            lh_loss = likelihood_loss(
                x_rec.view(x_rec.size(0), -1),
                data_tensor.view(data_tensor.size(0), -1),
                metric,
            )
            anomaly_score[:, i] = lh_loss.numpy()
            class_probs += F.softmax(class_logits, dim=1).numpy()

    avg_anomaly_score = np.mean(anomaly_score, axis=1)
    avg_class_probs = class_probs / num_sample

    # Combine reconstruction error and classification probability
    # For anomaly detection, you might consider:
    # - High reconstruction error
    # - High probability of anomaly class
    anomaly_scores = (
        avg_anomaly_score * avg_class_probs[:, 1]
    )  # Assuming class 1 is anomaly

    return anomaly_scores
