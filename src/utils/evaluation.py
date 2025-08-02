import torch
import numpy as np


def calc_centroid(model, data_loader):
    """
    Calculate class centroids for latent embeddings.
    Returns a dict: {class_label: centroid_vector}
    """
    class_embeddings = {i: [] for i in range(10)}
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to("cpu")
            if hasattr(model, 'get_latent_embedding'):
                latent = model.get_latent_embedding(images).cpu().numpy()
            else:
                latent = model(images).cpu().numpy()

            for emb, lbl in zip(latent, labels.cpu().numpy()):
                class_embeddings[lbl].append(emb)

    return {lbl: np.mean(embs, axis=0) for lbl, embs in class_embeddings.items()}


def test_accuracy(model, test_loader, device="cpu"):
    """
    Test model accuracy on clean examples.
    Returns accuracy in %.
    """
    model.to(device).eval()
    correct = total = 0

    with torch.no_grad():
        for _, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100.0 * correct / total


def get_latent_embedding(self, x):
    _ = self(x)  # forward pass to populate latent_output
    return latent_output.view(latent_output.size(0), -1)  # flatten avgpool output


def get_latent_hook(module, input, output):
    global latent_output
    latent_output = output
