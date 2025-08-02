import numpy as np
from cma import CMAEvolutionStrategy

from .base_attack import EvolutionaryAttack
from utils.fitness_functions import reward_function


class CMAESAttack(EvolutionaryAttack):
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) based attack.
    Serves as the baseline of the attack, with a varying initial population.
    """
    
    def _get_params(self, target_mode, config, dim):
        # CMA-ES parameters
        if target_mode:
            return {
                'popsize': 16,
                'bounds': [-config.epsilon, config.epsilon],
                'maxiter': 90,
                'seed': config.seed,
                'AdaptSigma': True,
                'CMA_active': True,
                'CMA_stds': config.epsilon * np.ones(dim),
                'tolx': 1e-6,
                'CMA_dampsvec_fac': 1.15,
                'verb_disp': 0
            }
        else:
            return {
                'popsize': 42,
                'bounds': [-config.epsilon, config.epsilon],
                'maxiter': 70,
                'seed': config.seed,
                'AdaptSigma': True,
                'CMA_active': False,
                'CMA_stds': config.epsilon * np.ones(dim),
                'tolx': 1e-6,
                'CMA_dampsvec_fac': 1.5,
                'verb_disp': 0
            }

    def _run_cmaes(self,
                   x_original: np.ndarray,
                   true_label: int,
                   initial_perturbations: np.ndarray,
                   target_mode: bool = True
                   ):
        """
        Run CMA-ES to find an adversarial example. As soon as one is found, stop and return it.
        Returns:
            adv_img:         the adversarial image (or last best if none found)
            best_pert:       the perturbation that produced the adversarial (or last best)
            iters_found:     number of model queries when found (or False if none found)
            best_pred_class: model's predicted class on the returned adv_img
        """
        self.queries        = 0
        self.target_queries = 0
        self.found          = False

        original_shape = x_original.shape # e.g. (1, 28, 28) for MNIST
        flat_dim       = int(np.prod(original_shape))  # e.g. 784

        # flatten the provided initial perturbations
        pop_size       = initial_perturbations.shape[0]
        init_pert_flat = initial_perturbations.reshape(pop_size, flat_dim)

        # use their mean as CMA-ES’s initial mean
        mean         = np.mean(init_pert_flat, axis=0)
        mean_clipped = np.clip(mean, -self.config.epsilon + 1e-8, self.config.epsilon - 1e-8)
        
        # get parameters & CMAES initialize attack
        cmaes_params = self._get_params(target_mode, self.config, flat_dim)

        es       = CMAEvolutionStrategy(mean_clipped, self.config.epsilon, cmaes_params)
        max_gens = cmaes_params['maxiter']

        # placeholder for the "best so far" in case we never find a true adversarial
        best_pert_flat  = mean_clipped.copy()
        best_fitness    = float('inf')
        best_adv_img    = x_original.copy()
        best_pred_class = self._get_prediction_class(self.target_model, x_original)

        for gen in range(max_gens):
            solutions = es.ask()

            # build all candidate images for this generation
            population_imgs = []
            for sol_flat in solutions:
                perturb       = sol_flat.reshape(original_shape)
                perturb       = self._clip_perturbation(perturb)
                adv_candidate = np.clip(x_original + perturb, 0, 1)
                population_imgs.append(adv_candidate)
            
            population_imgs = np.stack(population_imgs, axis=0)

            # evaluate each candidate in turn, but stop immediately if we find an adversarial
            fitness_vals = list()
            for idx, cand in enumerate(population_imgs):
                # we already found one within this generation, skip further candidates
                if self.found and target_mode:
                    fitness_vals.append(-self.bonus)
                    continue

                # query the target model
                pred = self._model_predict(self.target_model, cand)
                if target_mode: self.target_queries += 1

                # compute reward
                reward = reward_function(self, self.target_model, cand, pred, true_label, target_mode)

                # if reward == bonus, we found a valid adversarial, stop immediately
                if reward == self.bonus:
                    self.found = True

                    # save this adversarial candidate & perturbation
                    best_pert_flat  = solutions[idx]            # the flat perturbation vector
                    perturb_found   = solutions[idx].reshape(original_shape)
                    best_adv_img    = np.clip(x_original + self._clip_perturbation(perturb_found), 0, 1)
                    best_pred_class = int(np.argmax(pred))

                    iters = self.target_queries if target_mode else (gen + 1)
                    return best_adv_img, perturb_found, iters, best_pred_class

                fitness_vals.append(reward)

                # track best so far in case no adversarial is ever found
                if reward < best_fitness:
                    best_fitness   = reward
                    best_pert_flat = solutions[idx]

            # tell CMA-ES all fitness values (even if we didn’t find an adversarial this generation)
            es.tell(solutions, fitness_vals)

            # verbose logging
            if self.config.verbose:
                print(f"Generation {gen+1}/{max_gens} | Best fitness so far: {best_fitness:.5f}", end='\r')

        # if we exit all generations without finding a true adversarial, return "best-so-far"
        best_pert       = best_pert_flat.reshape(original_shape)
        best_pert       = self._clip_perturbation(best_pert)
        adv_img         = np.clip(x_original + best_pert, 0, 1)
        best_pred_class = self._get_prediction_class(self.target_model, adv_img)
        return adv_img, best_pert, False, best_pred_class
