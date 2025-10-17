import torch
import numpy as np


def calc_centroid(model, data_loader, num_classes, device=None):
    """
    Calculate class centroids for latent embeddings.
    Returns a dict: {class_label: centroid_vector}
    
    Args:
        model: The model to use for embedding extraction
        data_loader: DataLoader containing the data
        num_classes: Number of classes in the dataset
        device: Device to use for computations (if None, uses model's device)
    """
    
    # Determine device from model if not specified
    if device is None: device = next(model.parameters()).device
    
    class_embeddings = {i: [] for i in range(num_classes)}
    model.eval()
    
    with torch.no_grad():
        for images, labels in data_loader:
            # Move data to the same device as the model
            images = images.to(device)
            labels = labels.to(device)
            
            if hasattr(model, 'get_latent_embedding'):
                latent = model.get_latent_embedding(images).cpu().numpy()
            else:
                latent = model(images).cpu().numpy()

            for emb, lbl in zip(latent, labels.cpu().numpy()):
                class_embeddings[lbl].append(emb)

    return {lbl: np.mean(embs, axis=0) for lbl, embs in class_embeddings.items() if embs}  # Skip empty classes if any


def test_accuracy(model, data_loader, device=None):
    """
    Test model accuracy on a dataset.
    
    Args:
        model: The model to test
        data_loader: DataLoader containing test data
        device: Device to use for computations (if None, uses model's device)
    """

    # determine device from model if not specified
    if device is None:            device = next(model.parameters()).device
    elif isinstance(device, str): device = torch.device(device)
    
    model.eval()
    correct = 0
    total   = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            # Move data to the same device as the model
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def get_latent_embedding(self, x):
    _ = self(x)  # forward pass to populate latent_output
    return latent_output.view(latent_output.size(0), -1)  # flatten avgpool output


def get_latent_hook(module, input, output):
    global latent_output
    latent_output = output
