import torch
import numpy as np
import torch.nn.functional as F


def reward_function(self, model, adv_image, prediction, true_label, target):
    """
    Reward function for TGEA, defined in the 6.3.5 section of the thesis.
    Return a very large positive value (e.g. 999999999) if an adversarial is found (fooling). 
    Otherwise, compute: (α⋅max_conf + β⋅other_conf_avg) - γ⋅true_conf + δ⋅centroid_dist - ε⋅surrogate_loss, then return the reward.
    """

    # hyperparameters
    alpha = 0.4    # weight of adversarial maximum confidence
    beta = 0       # weight of average of other confidences
    gamma = 1.1    # weight of true confidence
    delta = 0.9    # weight of centroid distance

    if not target:
        adv_tensor = torch.tensor(adv_image, dtype=torch.float32, device="cpu").unsqueeze(0)
        latent_embedding = model.get_latent_embedding(adv_tensor).squeeze(0)

        # Compute distances to all centroids
        distances = {
            cls: torch.norm(latent_embedding - torch.tensor(centroid, device="cpu"))
            for cls, centroid in self.centroids.items()
        }
        original_distance = distances[true_label]
        other_classes = [cls for cls in self.centroids if cls != true_label]
        min_other = min(distances[cls] for cls in other_classes)
        centroid_dist = (original_distance - min_other).item()

    else:
        centroid_dist  = 0

    # extract true and other confidences from `prediction`
    true_conf   = prediction[0][true_label]
    other_confs = prediction[0].copy()

    # find max among “other” classes
    other_confs[true_label] = -float('inf')
    max_conf = max(other_confs)

    # compute average of all other confidences (set true label’s slot to 0)
    other_confs[true_label] = 0
    other_conf_avg = sum(other_confs) / (len(prediction[0]) - 1)

    # if we already fooled the model, return bonus
    if max_conf > true_conf:
        if target:
            return self.bonus
        elif not self.found:
            return self.bonus

    reward = (alpha * max_conf + beta * other_conf_avg) - gamma * true_conf + delta * centroid_dist
    return -reward


def reward_function_ga(self, ga_instance, solution, solution_idx):
    """
    Reward function used inside the GeneticAlgorithmAttack.
    Returns a scalar fitness score combining confidence margins, centroid distances, and a diversity term.
    """

    # hyperparameters
    alpha = 0.3
    beta  = 0.05
    gamma = 1.3
    delta = 0.9

    # build adversarial image: current_image + solution
    adv = self.current_image + solution.reshape(self.current_image.shape)
    adv = torch.tensor(adv, dtype=torch.float32, device="cpu").clamp(0, 1)

    self.queries += 1

    # surrogate model prediction
    with torch.no_grad():
        pred = self.surr_model(adv.unsqueeze(0)).softmax(dim=1).cpu().numpy()

    true_conf = pred[0][self.current_label]
    other_confs = pred[0].copy()
    other_confs[self.current_label] = -float('inf')
    max_conf = max(other_confs)
    other_confs[self.current_label] = 0
    other_conf_avg = sum(other_confs) / (len(pred[0]) - 1)

    # latent embedding distances
    latent_emb = self.surr_model.get_latent_embedding(adv.unsqueeze(0)).squeeze(0)
    distances = {
        cls: torch.norm(latent_emb - torch.tensor(centroid, device="cpu"))
        for cls, centroid in self.centroids.items()
    }
    original_distance = distances.get(self.current_label, torch.tensor(0.0))
    min_other = min(distances[cls] for cls in distances if cls != self.current_label) if self.centroids else torch.tensor(0.0)
    centroid_dist = (original_distance - min_other).item() if self.centroids else 0

    # diversity penalty across GA population
    diversity_term = 0
    for other_sol in ga_instance.population:
        sim = np.mean((solution - other_sol) ** 2)
        diversity_term += sim
    diversity_term = -diversity_term / len(ga_instance.population)
    diversity_term *= 22

    reward = (alpha * max_conf + beta * other_conf_avg) - gamma * true_conf + delta * centroid_dist
    reward += diversity_term

    return reward
