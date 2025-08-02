import time
import numpy as np
import pygad
import os

from .base_attack import EvolutionaryAttack
from utils.fitness_functions import reward_function_ga
from utils.visualization import save_comparison_image
from .cmaes_attack import CMAESAttack

class GeneticAlgorithmAttack(EvolutionaryAttack):
    """Genetic Algorithm based adversarial attack."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        raw_fitness = ga_instance.last_generation_fitness
        pop         = ga_instance.population
        sigma       = np.std(pop) * 0.5
        adjusted    = self._fitness_sharing(pop, raw_fitness, sigma)
        ga_instance.last_generation_fitness = adjusted

    def _custom_mutation(self, offspring: np.ndarray, ga_instance) -> np.ndarray:
        """
        Custom mutation: add uniform noise to a subset of genes, then clip.
        """
        for idx in range(offspring.shape[0]):
            mutation_indices = np.random.choice(
                range(offspring.shape[1]),
                size=int(ga_instance.mutation_percent_genes / 100.0 * offspring.shape[1]),
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

        self.surr_model = models
        self.centroids = self.config.centroids

        if not os.path.exists('./results'):
            os.makedirs('./results')
        
        results_file = f"./results/comparison_results_{self.config.dataset}.txt"

        for idx, (x_orig, y_true) in enumerate(zip(x_batch, y_batch)):
            if self.config.verbose:
                print(f"Processing image {idx + 1}/{len(x_batch)}...")

            # get the initial population of the genetic algorithm
            self.queries = 0

            np.random.seed(self.config.seed)
            init_pop = np.random.uniform(
                low=-self.config.epsilon,
                high=self.config.epsilon,
                size=(self.ga_config['population_size'], np.prod(x_orig.shape))
            )

            self.current_image = x_orig
            self.current_label = y_true

            # create ga instance
            ga_instance = pygad.GA(
                num_generations=self.ga_config['generations'],
                num_parents_mating=self.ga_config['num_parents_mating'],
                fitness_func=lambda ga_instance, solution, sol_idx: reward_function_ga(
                    self, ga_instance, solution, sol_idx
                ),
                initial_population=init_pop,
                num_genes=int(np.prod(x_orig.shape)),
                mutation_type=self._custom_mutation,
                crossover_type="single_point",
                parent_selection_type="sss",
                on_generation=self._fitness_sharing_callback
            )

            ga_instance.run()

            final_pop      = ga_instance.population
            fitness_scores = ga_instance.last_generation_fitness

            sorted_pop = sorted(zip(final_pop, fitness_scores), key=lambda x: x[1], reverse=True)

            top_sols = [sol.reshape(x_orig.shape) for sol, _ in sorted_pop[:top_n]]

            # initialize CMA-ES with top GA solutions (with a little gaussian noise)
            top_arr  = np.stack(top_sols, axis=0)
            top_arr += np.random.normal(0, 0.035, top_arr.shape)
            top_arr  = np.clip(top_arr, -self.config.epsilon, self.config.epsilon)

            cmaes_att = CMAESAttack(self.target_model, self.surrogate_model, self.config)
            adv_example, _, iterations, prediction = cmaes_att._run_cmaes(
                x_orig, y_true, top_arr
            )

            total_queries += self.queries
            adversarial_examples.append(adv_example)

            if self.config.verbose:
                cls_name = (self.config.class_names[prediction]
                            if prediction < len(self.config.class_names)
                            else str(prediction))
                true_name = (self.config.class_names[y_true]
                             if y_true < len(self.config.class_names)
                             else str(y_true))
                print(f"Queries: {iterations} | Pred: {cls_name} | True: {true_name}\n")
            
            # try random population
            cmaes_att = CMAESAttack(self.target_model, self.surrogate_model, self.config)
            rand_pop = np.random.uniform(-self.config.epsilon, self.config.epsilon, (top_n,) + self.current_image.shape)
            adv_example, _, r_iterations, prediction = cmaes_att._run_cmaes(
                x_orig, y_true, rand_pop
            )

            with open(results_file, "a") as f:
                print("Writing")
                f.write(f"<{iterations}, {r_iterations}>\n")

        return total_queries, adversarial_examples
