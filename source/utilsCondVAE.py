import torch
import math
import time
import numpy as np
from torch.autograd import Variable
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
class Dataset(torch.utils.data.Dataset):
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



def get_dataset(train, validation, batch_size=128):
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

    train_data = torch.utils.data.DataLoader(train, batch_size=batch_size, pin_memory=cuda,
                                             sampler=get_sampler(train.data))
    valid_data = torch.utils.data.DataLoader(validation, batch_size=batch_size, pin_memory=cuda,
                                             sampler=get_sampler(validation.data))
    
    return train_data, valid_data


def likelihood_loss(r, x, metric='BCE'): #ChatGPT(#CondVAE): Since the data might not be binary, consider using Mean Squared Error (MSE) loss.
    """calculates likelihood loss between input and its reconstruction

    Args:
        r (tensor): reconstructed data
        x (tensor): input data

    Returns:
        tensor: likelihood loss between reconstructed and input data
    """
    r = r.view(r.size()[0], -1)
    x = x.view(x.size()[0], -1)
    likelihood_loss = -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)

    if metric == 'BCE':
        likelihood_loss = -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)
    elif metric == 'MSE':
        mse_loss = torch.nn.MSELoss(reduction='none')
        likelihood_loss = torch.sum(mse_loss(x, r), dim=-1)

    return likelihood_loss


def train_model(model, optimizer, model_name, train_labeled_data, train_unlabeled_data, valid_data, save_dir,
                metric='MSE', beta=1, num_epochs=100, save=True, verbose=1):
    """Trains the semi-supervised VAE model."""
    model = model.to(device) 
    print("Using device:", device)
    training_total_loss = np.zeros(num_epochs)
    training_rec_loss = np.zeros(num_epochs)
    training_kl_loss = np.zeros(num_epochs)
    training_class_loss = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        model.train()
        total_loss, rec_loss, kl_loss, class_loss = (0, 0, 0, 0)
        
        # Training with labeled data
        for x, y in train_labeled_data:
            x = x.to(device)
            y = y.to(device)
            y_onehot = F.one_hot(y, num_classes=model.num_classes).float()

            x_rec, class_logits = model(x, y_onehot)

            # Reconstruction loss
            likelihood = -likelihood_loss(x_rec, x, metric)
            rec_loss_batch = torch.mean(-likelihood)

            # KL divergence
            kl_div_batch = torch.mean(model.kl_div)

            # Classification loss
            class_loss_batch = F.cross_entropy(class_logits, y)

            # Total loss
            L = -torch.mean(likelihood - beta * model.kl_div) + class_loss_batch

            # Backpropagation
            optimizer.zero_grad()
            L.backward()
            optimizer.step()

            # Accumulate losses
            rec_loss += rec_loss_batch.item()
            kl_loss += kl_div_batch.item()
            class_loss += class_loss_batch.item()
            total_loss += L.item()

        # Training with unlabeled data
        for x in train_unlabeled_data:
            x = x.to(device)
            x_rec, class_logits = model(x)

            # Use predicted labels
            y_onehot = F.softmax(class_logits, dim=1)

            # Reconstruction loss
            likelihood = -likelihood_loss(x_rec, x, metric)
            rec_loss_batch = torch.mean(-likelihood)

            # KL divergence
            kl_div_batch = torch.mean(model.kl_div)

            # Total loss (no classification loss)
            L = -torch.mean(likelihood - beta * model.kl_div)

            # Backpropagation
            optimizer.zero_grad()
            L.backward()
            optimizer.step()

            # Accumulate losses
            rec_loss += rec_loss_batch.item()
            kl_loss += kl_div_batch.item()
            total_loss += L.item()

        # Logging
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Total Loss: {total_loss:.4f}, Rec Loss: {rec_loss:.4f}, KL Loss: {kl_loss:.4f}, Class Loss: {class_loss:.4f}")

    if save:
        torch.save(model.state_dict(), f"{save_dir}/{model_name}.pth")
    return model


def find_score(model, data, metric='MSE', num_sample=50):
    model = model.to('cpu')
    model.eval()
    num_data = np.shape(data)[0]
    data_tensor = torch.tensor(data).float()
    anomaly_score = np.zeros((num_data, num_sample))
    class_probs = np.zeros((num_data, model.num_classes))

    with torch.no_grad():
        for i in range(num_sample):
            x_rec, class_logits = model(data_tensor)
            lh_loss = likelihood_loss(x_rec.view(x_rec.size(0), -1),
                                      data_tensor.view(data_tensor.size(0), -1),
                                      metric)
            anomaly_score[:, i] = lh_loss.numpy()
            class_probs += F.softmax(class_logits, dim=1).numpy()

    avg_anomaly_score = np.mean(anomaly_score, axis=1)
    avg_class_probs = class_probs / num_sample

    # Combine reconstruction error and classification probability
    # For anomaly detection, you might consider:
    # - High reconstruction error
    # - High probability of anomaly class
    anomaly_scores = avg_anomaly_score * avg_class_probs[:, 1]  # Assuming class 1 is anomaly

    return anomaly_scores

