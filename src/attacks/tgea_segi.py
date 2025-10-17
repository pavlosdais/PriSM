import time
import numpy as np
import pygad
import os
import torch

from .base_attack import EvolutionaryAttack
from utils.fitness_functions import reward_function_ga
from utils.visualization import save_comparison_image
from .cmaes import CMAESAttack

class GeneticAlgorithmAttack(EvolutionaryAttack):
    """Genetic Algorithm based adversarial attack with GPU support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # set up device
        if hasattr(self.config, 'device'):
            if isinstance(self.config.device, str): self.device = torch.device(self.config.device)
            else:                                  self.device = self.config.device
        else: self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
        self.ga_config = {
            'population_size': 60,
            'generations': 50,
            'num_parents_mating': 5,
            'mutation_percent_genes': 10
        }

    def _fitness_sharing(self,
                         population: np.ndarray,
                         fitness_values: np.ndarray,
                         sharing_sigma: float) -> np.ndarray:
        """
        Apply fitness sharing to encourage diversity.
        """

        num_ind = population.shape[0]
        adjusted = np.zeros_like(fitness_values)

        for i in range(num_ind):
            share_sum = 0
            for j in range(num_ind):
                if i != j:
                    dist = np.linalg.norm(population[i] - population[j])
                    if dist < sharing_sigma:
                        share_sum += 1 - (dist / sharing_sigma)
            adjusted[i] = fitness_values[i] / (1 + share_sum)

        return adjusted

    def _fitness_sharing_callback(self, ga_instance):
        """
        PyGAD callback to apply fitness sharing at each generation.
        """

        try:
            raw_fitness = ga_instance.last_generation_fitness
            pop = ga_instance.population
            sigma = np.std(pop) * 0.5
            
            # avoid division by zero
            if sigma == 0: sigma = 0.001
                
            adjusted = self._fitness_sharing(pop, raw_fitness, sigma)
            ga_instance.last_generation_fitness = adjusted
        except Exception as e:
            print(f"Warning: Fitness sharing failed: {e}")

    def _custom_mutation(self, offspring: np.ndarray, ga_instance) -> np.ndarray:
        """
        Custom mutation: add uniform noise to a subset of genes, then clip.
        """

        try:
            for idx in range(offspring.shape[0]):
                # calculate number of genes to mutate
                num_genes_to_mutate = int(ga_instance.mutation_percent_genes / 100.0 * offspring.shape[1])
                if num_genes_to_mutate == 0:
                    num_genes_to_mutate = 1
                
                mutation_indices = np.random.choice(
                    range(offspring.shape[1]),
                    size=min(num_genes_to_mutate, offspring.shape[1]),
                    replace=False
                )
                
                for gene_idx in mutation_indices:
                    val = np.random.uniform(-self.config.epsilon, self.config.epsilon)
                    offspring[idx, gene_idx] += val
                    offspring[idx, gene_idx] = np.clip(
                        offspring[idx, gene_idx],
                        -self.config.epsilon,
                        self.config.epsilon
                    )
        except Exception as e:
            print(f"Warning: Custom mutation failed: {e}")
            
        return offspring

    def run_attack(self,
                    x_batch: np.ndarray,
                    y_batch: np.ndarray,
                    models,
                    top_n: int = 13):
        """
        Run GA attack on each (x, y) in the batch. Returns (total_queries, [adv_examples]).
        """

        adversarial_examples = []
        total_queries = 0

        # Ensure models are on the correct device
        if self.target_model is not None:    self.target_model    = self.target_model.to(self.device)
        if self.surrogate_model is not None: self.surrogate_model = self.surrogate_model.to(self.device)
            
        self.surr_model = models
        if models is not None:
            if hasattr(models, 'to'):
                self.surr_model = models.to(self.device)
            
        self.centroids = self.config.centroids

        if not os.path.exists('./results'): os.makedirs('./results')
        results_file = f"./results/comparison_results_{self.config.dataset}.txt"

        for idx, (x_orig, y_true) in enumerate(zip(x_batch, y_batch)):
            if self.config.verbose:
                print(f"Processing image {idx + 1}/{len(x_batch)}...")

            try:
                # convert tensors to numpy if necessary
                if isinstance(x_orig, torch.Tensor): x_orig_np = x_orig.detach().cpu().numpy()
                else:                                x_orig_np = x_orig
                    
                if isinstance(y_true, torch.Tensor):
                    y_true_np = y_true.detach().cpu().numpy()
                    if isinstance(y_true_np, np.ndarray) and y_true_np.ndim > 0:
                        y_true_np = y_true_np.item()
                else:
                    y_true_np = y_true

                self.queries = 0
                np.random.seed(self.config.seed + idx)
                
                # create initial population
                init_pop = np.random.uniform(
                    low=-self.config.epsilon,
                    high=self.config.epsilon,
                    size=(self.ga_config['population_size'], np.prod(x_orig_np.shape))
                ).astype(np.float32)

                self.current_image = x_orig_np
                self.current_label = y_true_np

                # create and run GA instance
                ga_instance = pygad.GA(
                    num_generations=int(self.ga_config['generations'] / 2),
                    num_parents_mating=self.ga_config['num_parents_mating'],
                    fitness_func=lambda ga_instance, solution, sol_idx: reward_function_ga(
                        self, ga_instance, solution, sol_idx
                    ),
                    initial_population=init_pop,
                    num_genes=int(np.prod(x_orig_np.shape)),
                    mutation_type=self._custom_mutation,
                    crossover_type="single_point",
                    parent_selection_type="sss",
                    on_generation=self._fitness_sharing_callback,
                    stop_criteria="reach_1.0"
                )

                ga_instance.run()

                # get final population and fitness scores
                final_pop      = ga_instance.population
                fitness_scores = ga_instance.last_generation_fitness

                # sort population by fitness (descending order)
                sorted_pop = sorted(zip(final_pop, fitness_scores), key=lambda x: x[1], reverse=True)

                # get top solutions
                top_sols = [sol.reshape(x_orig_np.shape) for sol, _ in sorted_pop[:top_n]]

                # initialize CMA-ES with top GA solutions (with a little gaussian noise)
                top_arr  = np.stack(top_sols, axis=0).astype(np.float32)
                top_arr += np.random.normal(0, 0.035, top_arr.shape).astype(np.float32)
                top_arr  = np.clip(top_arr, -self.config.epsilon, self.config.epsilon)

                # run CMA-ES refinement with GA-evolved solutions
                cmaes_att = CMAESAttack(self.target_model, self.surrogate_model, self.config)
                adv_example, _, iterations, prediction = cmaes_att._run_cmaes(
                    x_orig_np, y_true_np, top_arr
                )

                total_queries += self.queries
                adversarial_examples.append(adv_example)

                if self.config.verbose:
                    if hasattr(self.config, 'class_names') and self.config.class_names:
                        cls_name = (self.config.class_names[prediction]
                                    if prediction < len(self.config.class_names)
                                    else str(prediction))
                        true_name = (self.config.class_names[y_true_np]
                                     if y_true_np < len(self.config.class_names)
                                     else str(y_true_np))
                    else:
                        cls_name = str(prediction)
                        true_name = str(y_true_np)
                    
                    print(f"Queries: {iterations} | Pred: {cls_name} | True: {true_name}")
                
                # try random population for comparison
                try:
                    cmaes_att_random = CMAESAttack(self.target_model, self.surrogate_model, self.config)
                    rand_pop = np.random.uniform(
                        -self.config.epsilon, 
                        self.config.epsilon, 
                        (top_n,) + x_orig_np.shape
                    ).astype(np.float32)
                    
                    adv_example_rand, _, r_iterations, prediction_rand = cmaes_att_random._run_cmaes(
                        x_orig_np, y_true_np, rand_pop
                    )

                    # save comparison results
                    with open(results_file, "a") as f:
                        f.write(f"<{iterations}, {r_iterations}>\n")
                        
                except Exception as e:
                    print(f"Warning: Random population CMA-ES failed: {e}")
                    with open(results_file, "a") as f:
                        f.write(f"<{iterations}, -1>\n")

                # save comparison image if possible
                try:
                    save_comparison_image(x_orig_np, adv_example, f"{self.config.dataset}_{idx}", self.config.dataset)
                except Exception as e:
                    print(f"Warning: Could not save comparison image: {e}")

            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                import traceback
                traceback.print_exc()
                
                adversarial_examples.append(None)
                
                try:
                    with open(results_file, "a") as f:
                        f.write(f"<-1, -1>\n")
                except: pass
                
                continue

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return total_queries, adversarial_examples
