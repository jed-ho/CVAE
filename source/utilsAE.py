import torch
import math
import time
import numpy as np
from torch.autograd import Variable

cuda = torch.cuda.is_available()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        X = self.data[index, :, :]
        return X

def get_dataset(train, validation, batch_size=128):
    """Creates the dataloaders for training and validation data.

    Args:
        train (Dataset): Training dataset.
        validation (Dataset): Validation dataset.
        batch_size (int, optional): Batch size for training. Defaults to 128.

    Returns:
        tuple: Training and validation dataloaders.
    """
    from torch.utils.data.sampler import SubsetRandomSampler

    def get_sampler(data):
        num_data = data.shape[0]
        sampler = SubsetRandomSampler(np.arange(0, num_data))
        return sampler

    train_data = torch.utils.data.DataLoader(
        train, batch_size=batch_size, pin_memory=cuda, sampler=get_sampler(train.data)
    )
    valid_data = torch.utils.data.DataLoader(
        validation, batch_size=batch_size, pin_memory=cuda, sampler=get_sampler(validation.data)
    )
    
    return train_data, valid_data

def reconstruction_loss(r, x, metric='MSE'):
    """Calculates reconstruction loss between input and its reconstruction.

    Args:
        r (Tensor): Reconstructed data.
        x (Tensor): Input data.
        metric (str, optional): Loss metric to use ('MSE' or 'BCE'). Defaults to 'MSE'.

    Returns:
        Tensor: Reconstruction loss between reconstructed and input data.
    """
    r = r.view(r.size(0), -1)
    x = x.view(x.size(0), -1)

    if metric == 'BCE':
        loss_fn = torch.nn.BCELoss(reduction='sum')
        rec_loss = loss_fn(r, x)
    elif metric == 'MSE':
        loss_fn = torch.nn.MSELoss(reduction='sum')
        rec_loss = loss_fn(r, x)
    else:
        raise ValueError("Unsupported metric. Use 'MSE' or 'BCE'.")

    return rec_loss

def train_model(model, optimizer, model_name, train_data, valid_data, save_dir,
                metric='MSE', num_epochs=100, save=True, verbose=1):
    """Trains the autoencoder model.

    Args:
        model (nn.Module): Autoencoder model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        model_name (str): Name for saving the model.
        train_data (DataLoader): Training dataloader.
        valid_data (DataLoader): Validation dataloader.
        save_dir (str): Directory to save the model.
        metric (str, optional): Loss metric ('MSE' or 'BCE'). Defaults to 'MSE'.
        num_epochs (int, optional): Number of training epochs. Defaults to 100.
        save (bool, optional): Whether to save the trained model. Defaults to True.
        verbose (int, optional): Verbosity level. Defaults to 1.

    Returns:
        nn.Module: Trained model.
    """
    training_loss = np.zeros(num_epochs)  # Records average loss per epoch

    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        total_loss = 0
        total_samples = 0

        for x in train_data:
            x = Variable(x)
            if cuda:
                x = x.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            reconstruction = model(x)

            # Compute loss
            loss = reconstruction_loss(reconstruction, x, metric)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss and sample count
            total_loss += loss.item()
            total_samples += x.size(0)
        
        # Calculate average training loss
        avg_train_loss = total_loss / total_samples
        training_loss[epoch] = avg_train_loss

        # Validation phase
        model.eval()
        valid_loss = 0
        valid_samples = 0

        with torch.no_grad():
            for x_val in valid_data:
                x_val = Variable(x_val)
                if cuda:
                    x_val = x_val.cuda()

                reconstruction_val = model(x_val)
                loss_val = reconstruction_loss(reconstruction_val, x_val, metric)
                valid_loss += loss_val.item()
                valid_samples += x_val.size(0)

        avg_valid_loss = valid_loss / valid_samples

        end = time.time()
        duration = end - start

        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}], Duration: {duration:.2f}s, "
                  f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}")

    if save:
        torch.save(model.state_dict(), save_dir + model_name + ".pth")
        np.savez_compressed(save_dir + model_name + "_training_loss.npz", training_loss=training_loss)

    return model

def find_score(model, data, metric='MSE'):
    """Calculates reconstruction errors for anomaly detection.

    Args:
        model (nn.Module): Trained autoencoder model.
        data (ndarray): Input data.
        metric (str, optional): Loss metric ('MSE' or 'BCE'). Defaults to 'MSE'.

    Returns:
        ndarray: Reconstruction error for each data point.
    """
    model.eval()
    data_tensor = torch.tensor(data).float()
    if cuda:
        data_tensor = data_tensor.cuda()

    with torch.no_grad():
        reconstruction = model(data_tensor)
        r = reconstruction.reshape(reconstruction.size(0), -1)
        x = data_tensor.reshape(data_tensor.size(0), -1)

        if metric == 'BCE':
            loss_fn = torch.nn.BCELoss(reduction='none')
        elif metric == 'MSE':
            loss_fn = torch.nn.MSELoss(reduction='none')
        else:
            raise ValueError("Unsupported metric. Use 'MSE' or 'BCE'.")

        rec_error = loss_fn(r, x)
        rec_error = rec_error.sum(dim=1).cpu().numpy()

    return rec_error
