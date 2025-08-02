import torch
import numpy as np

from attacks.genetic_attack import GeneticAlgorithmAttack
from attacks.cmaes_attack import CMAESAttack
from attacks.sgsa_init import SGSQInitialization
from attacks.mixed_attack import TransferAttackSeededInitialization
from attacks.config import AdversarialAttackConfig

def test_attack(model, surr_model,
               loader, class_centroids=None,
               dataset_name="mnist", batches_skip=0,
               attack_type = "TASI", verbose=True,
               epsilon_v=0.2):
    """
    Run adversarial-attack test using CMA-ES/GA/MixedMethod.
    Returns total number of images processed.
    """
    device = "cpu"
    model.to(device).eval()
    if surr_model is not None:
        surr_model.to(device).eval()

    total_tgea = total_sq = num_tgea = num_sq = total_img = batches = 0

    # attack configuration
    config = AdversarialAttackConfig(
        epsilon=epsilon_v,
        max_iter=1000,
        population_size=250,
        dataset=dataset_name,
        device=device,
        seed=42,
        verbose=verbose,
        centroids=class_centroids
    )

    for images, labels in loader:
        batches += 1

        if batches <= batches_skip:
            continue

        print(f"At batch = {batches} | attack type = {attack_type}")

        images_np = images.cpu().numpy() if isinstance(images, torch.Tensor) else images
        labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

        total_img += len(images_np)

        try:
            # Saliency Guided Square Attack
            if attack_type == "sgsa":
                attack = SGSQInitialization(model, surr_model, config)
                total_queries, total_square_queries, examples, examples_square = \
                    attack.run_attack(images_np, labels_np)
            
            # initialization method 1: Transfer Attack Seeded Initialization (TASI)
            elif attack_type == "tasi":
                attack = TransferAttackSeededInitialization(model, surr_model, config)
                total_queries, total_square_queries, examples, examples_square = \
                    attack.run_attack(images_np, labels_np)

            # initialization method 2: Surrogate-Evolved Genetic Initialization (SEGI)
            else:
                attack = GeneticAlgorithmAttack(model, surr_model, config)
                total_queries, adversarial_examples = \
                    attack.run_attack(images_np, labels_np, surr_model)
                examples = len([ex for ex in adversarial_examples if ex is not None])
                total_square_queries = examples_square = 0

            # ipdate aggregated stats
            num_tgea   += examples
            total_tgea += total_queries
            num_sq     += examples_square
            total_sq   += total_square_queries

            # print intermediate progress
            if num_tgea > 0 and num_sq > 0:
                avg_tgea = total_tgea / num_tgea
                avg_sq   = total_sq / num_sq
                sr_tgea  = (num_tgea / total_img) * 100
                sr_sq    = (num_sq / total_img) * 100
                print(f"Avg Queries TGEA= {avg_tgea:.2f}, Avg Queries Square= {avg_sq:.2f}, "
                      f"TGEA success= {sr_tgea:.2f}%, Square success= {sr_sq:.2f}%")
            elif num_tgea > 0:
                avg_tgea = total_tgea / num_tgea
                sr_tgea  = (num_tgea / total_img) * 100
                print(f"Avg Queries TGEA= {avg_tgea:.2f}, TGEA success= {sr_tgea:.2f}%")
            else:
                print("No successful attacks found yet.")

        except Exception as e:
            print(f"Error in batch {batches}: {e}")
            continue

    # final summary
    if num_tgea > 0:
        final_avg_tgea = total_tgea / num_tgea
        final_sr_tgea  = (num_tgea / total_img) * 100
        print("\nFinal Results:")
        print(f"TGEA — Average Queries: {final_avg_tgea:.2f}, Success Rate: {final_sr_tgea:.2f}%")

        if num_sq > 0:
            final_avg_sq = total_sq / num_sq
            final_sr_sq = (num_sq / total_img) * 100
            print(f"Square — Average Queries: {final_avg_sq:.2f}, Success Rate: {final_sr_sq:.2f}%")

    return total_img
