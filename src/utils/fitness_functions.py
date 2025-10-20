import torch
import numpy as np
import torch.nn.functional as F


def reward_function(self, model, adv_image, prediction, true_label, target):
    """
    Reward function, defined in the 6.3.5 section of the thesis.
    Return a very large positive value (e.g. 999999999) if an adversarial is found (fooling). 
    Otherwise, compute: (α⋅max_conf + β⋅other_conf_avg) - γ⋅true_conf + δ⋅centroid_dist - ε⋅surrogate_loss, then return the reward.
    """

    # hyperparameters
    alpha = 0.4    # weight of adversarial maximum confidence
    beta  = 0      # weight of average of other confidences
    gamma = 1.1    # weight of true confidence
    delta = 0.9    # weight of centroid distance

    if not target:
        # determine device from model
        device = next(model.parameters()).device
        
        # convert adv_image to tensor on the correct device
        if isinstance(adv_image, np.ndarray): adv_tensor = torch.tensor(adv_image, dtype=torch.float32, device=device)
        else:                                 adv_tensor = adv_image.to(device)

        if adv_tensor.dim() == 3: adv_tensor = adv_tensor.unsqueeze(0)
        try:
            with torch.no_grad():
                latent_embedding = model.get_latent_embedding(adv_tensor).squeeze(0)

            # compute distances to all centroids
            distances = {}
            for cls, centroid in self.centroids.items():
                if isinstance(centroid, np.ndarray):
                    centroid_tensor = torch.tensor(centroid, dtype=torch.float32, device=device)
                else:
                    centroid_tensor = centroid.to(device)
                distances[cls] = torch.norm(latent_embedding - centroid_tensor)

            original_distance = distances[true_label]
            other_classes = [cls for cls in self.centroids if cls != true_label]
            if other_classes:
                min_other = min(distances[cls] for cls in other_classes)
                centroid_dist = (original_distance - min_other).item()
            else:
                centroid_dist = 0
        except Exception as e:
            print(f"Warning: Centroid distance calculation failed: {e}")
            centroid_dist = 0
    else:
        centroid_dist = 0

    # extract true and other confidences from `prediction`
    true_conf = prediction[0][true_label]
    other_confs = prediction[0].copy()

    # find max among "other" classes
    other_confs[true_label] = -float('inf')
    max_conf = max(other_confs)

    # compute average of all other confidences (set true label's slot to 0)
    other_confs[true_label] = 0
    other_conf_avg = sum(other_confs) / (len(prediction[0]) - 1)

    # we already fooled the model, return bonus
    if max_conf > true_conf:
        if target:           return self.bonus
        elif not self.found: return self.bonus

    reward = (alpha * max_conf + beta * other_conf_avg) - gamma * true_conf + delta * centroid_dist
    return -reward


def reward_function_ga(self, ga_instance, solution, solution_idx):
    """
    Reward function used inside the GeneticAlgorithmAttack.
    Returns a scalar fitness score combining confidence margins, centroid distances, and a diversity term.
    """

    # hyperparameters
    alpha = 0.3
    beta = 0.05
    gamma = 1.3
    delta = 0.9

    try:
        # determine device from surrogate model
        device = next(self.surr_model.parameters()).device
        
        # build adversarial image: current_image + solution
        adv = self.current_image + solution.reshape(self.current_image.shape)
        
        # convert to tensor on the correct device and clamp
        adv_tensor = torch.tensor(adv, dtype=torch.float32, device=device).clamp(0, 1)
        
        # ensure correct dimensions for model input
        if adv_tensor.dim() == 3:
            adv_tensor = adv_tensor.unsqueeze(0)

        self.queries += 1

        # surrogate model prediction
        with torch.no_grad():
            logits = self.surr_model(adv_tensor)
            pred = F.softmax(logits, dim=1).cpu().numpy()

        true_conf = pred[0][self.current_label]
        other_confs = pred[0].copy()
        other_confs[self.current_label] = -float('inf')
        max_conf = max(other_confs)
        other_confs[self.current_label] = 0
        other_conf_avg = sum(other_confs) / (len(pred[0]) - 1)

        # latent embedding distances (if centroids are available)
        centroid_dist = 0
        if hasattr(self, 'centroids') and self.centroids:
            try:
                latent_emb = self.surr_model.get_latent_embedding(adv_tensor).squeeze(0)
                distances = {}
                
                for cls, centroid in self.centroids.items():
                    if isinstance(centroid, np.ndarray):
                        centroid_tensor = torch.tensor(centroid, dtype=torch.float32, device=device)
                    else:
                        centroid_tensor = centroid.to(device)
                    distances[cls] = torch.norm(latent_emb - centroid_tensor)

                original_distance = distances.get(self.current_label, torch.tensor(0.0, device=device))
                other_classes = [cls for cls in distances if cls != self.current_label]
                
                if other_classes:
                    min_other = min(distances[cls] for cls in other_classes)
                    centroid_dist = (original_distance - min_other).item()
                    
            except Exception as e:
                print(f"Warning: Centroid distance calculation failed in GA: {e}")
                centroid_dist = 0

        # diversity penalty across GA population
        diversity_term = 0
        try:
            for other_sol in ga_instance.population:
                sim = np.mean((solution - other_sol) ** 2)
                diversity_term += sim
            diversity_term = -diversity_term / len(ga_instance.population)
            diversity_term *= 22
        except Exception as e:
            print(f"Warning: Diversity term calculation failed: {e}")
            diversity_term = 0

        reward = (alpha * max_conf + beta * other_conf_avg) - gamma * true_conf + delta * centroid_dist
        reward += diversity_term

        return reward
        
    except Exception as e:
        print(f"Error in reward_function_ga: {e}")
        return -1000.0

def reward_function_cmaes(self, solution, x_original, true_label):
    """
    Reward function for CMA-ES optimization.
    """
    try:
        # determine device from target model
        device = next(self.target_model.parameters()).device
        
        # reshape solution to image shape and create adversarial example
        perturbation = solution.reshape(x_original.shape)
        adv_image = np.clip(x_original + perturbation, 0, 1)
        
        # convert to tensor on correct device
        adv_tensor = torch.tensor(adv_image, dtype=torch.float32, device=device)
        if adv_tensor.dim() == 3:
            adv_tensor = adv_tensor.unsqueeze(0)
        
        # get prediction from target model
        with torch.no_grad():
            logits = self.target_model(adv_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        
        # calculate reward based on confidence
        true_conf = probs[0][true_label]
        other_confs = probs[0].copy()
        other_confs[true_label] = -float('inf')
        max_conf = max(other_confs)
        
        # return high reward if attack is successful
        if max_conf > true_conf: return 1000.0
        
        # Otherwise return negative of true confidence (we want to minimize it)
        return -true_conf
        
    except Exception as e:
        print(f"Error in reward_function_cmaes: {e}")
        return -1000.0
