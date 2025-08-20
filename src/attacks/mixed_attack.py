import os
import numpy as np
import warnings

import torch
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import SquareAttack, ProjectedGradientDescent, BoundaryAttack

from .base_attack import EvolutionaryAttack
from utils.visualization import save_comparison_image
from .cmaes_attack import CMAESAttack

bound = True
if bound:
    import art.attacks.evasion.boundary as _boundary_module

    _orig_orth = _boundary_module.BoundaryAttack._orthogonal_perturb

    def _safe_orthogonal_perturb(self, delta, current_sample, original_sample):
        max_delta = 10.0
        delta = np.minimum(delta, max_delta)
        with np.errstate(over='ignore', invalid='ignore'):
            return _orig_orth(self, delta, current_sample, original_sample)

    _boundary_module.BoundaryAttack._orthogonal_perturb = _safe_orthogonal_perturb

class TransferAttackSeededInitialization(EvolutionaryAttack):
    """TASI attack combining Square Attack, PGD, Boundary"""

    def _create_art_classifier(self, model):
        """
        Wrap a PyTorch nn.Module into ART's PyTorchClassifier to run attacks.
        """
        return PyTorchClassifier(
            model=model,
            clip_values=(0, 1),
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
            input_shape=self.config.input_shape,
            nb_classes=self.config.num_classes,
            device_type=self.config.device
        )

    def _run_square_attack(self, classifier: PyTorchClassifier, image: np.ndarray) -> int:
        """
        Run a black-box SquareAttack on `classifier`, returning the query count.
        `image` must have shape (C, H, W). We expand to (1, C, H, W) for ART.
        """
        query_count = 0

        # wrap predict to count queries
        orig_predict = classifier.predict

        def counting_predict(x, batch_size=1):
            nonlocal query_count
            query_count += x.shape[0]
            return orig_predict(x.astype(np.float32), batch_size=batch_size)

        classifier.predict = counting_predict

        x_in = np.expand_dims(image.astype(np.float32), axis=0)

        sq = SquareAttack(
            estimator=classifier,
            norm=np.inf,
            eps=self.config.epsilon,
            max_iter=1001,
            batch_size=1,
            verbose=False
        )
        _ = sq.generate(x=x_in)

        # restore original predict
        classifier.predict = orig_predict
        return query_count

    def _generate_initial_population(self, image: np.ndarray, classifier: PyTorchClassifier, population_size: int = 12):
        """
        Combine Boundary, PGD, and Square seeds to form an initial population.
        Each returned noise has shape (C, H, W), clipped to ±ε.
        """
        initial_population = []
        q = population_size // 3

        # attack 1: Boundary Attack
        try:
            x_batch = np.expand_dims(image, axis=0).astype(np.float32)

            pgd_init = ProjectedGradientDescent(
                estimator=classifier,
                eps=self.config.epsilon,
                eps_step=(2/255 if self.config.dataset == "cifar10" else 0.01),
                max_iter=(50 if self.config.dataset == "cifar10" else 50),
                verbose=False
            )
            adv_init = pgd_init.generate(x=x_batch)
            if adv_init is None:
                adv_init = x_batch + np.random.uniform(-self.config.epsilon, self.config.epsilon, x_batch.shape)
            ba = BoundaryAttack(
                estimator=classifier,
                targeted=False,
                verbose=False,
                epsilon=self.config.epsilon,
                batch_size=1,
                max_iter=(360 if self.config.dataset == "cifar10" else 1200)
            )
            adv_bnd = ba.generate(x=x_batch, x_adv_init=adv_init)
            
            if adv_bnd is None: adv_img = adv_init[0]
            else: adv_img = adv_bnd[0]

            for _ in range(q):
                noise  = adv_img - image
                noise += np.random.normal(0, self.config.epsilon / 4, size=noise.shape)
                initial_population.append(self._clip_perturbation(noise))

        except Exception as e:
            warnings.warn(f"Boundary attack failed (falling back to random): {e}")
            for _ in range(q):
                noise = np.random.uniform(-self.config.epsilon, self.config.epsilon, image.shape)
                initial_population.append(noise)

        # attack 2: PGD
        try:
            x_batch = np.expand_dims(image, axis=0).astype(np.float32)
            pgd = ProjectedGradientDescent(
                estimator=classifier,
                eps=self.config.epsilon,
                eps_step=(0.005 if self.config.dataset == "cifar10" else 0.01),
                max_iter=(130 if self.config.dataset == "cifar10" else 100),
                verbose=False
            )
            for _ in range(q):
                init_noise = np.random.uniform(-self.config.epsilon / 3, self.config.epsilon / 3, image.shape)
                noisy      = np.clip(image + init_noise, 0, 1)
                adv_pgd    = pgd.generate(x=np.expand_dims(noisy.astype(np.float32), axis=0))
                if adv_pgd is not None:
                    noise = adv_pgd[0] - image
                else:
                    noise = noisy - image
                initial_population.append(self._clip_perturbation(noise))

        except Exception as e:
            warnings.warn(f"PGD attack failed (falling back to random): {e}")
            for _ in range(q):
                noise = np.random.uniform(-self.config.epsilon, self.config.epsilon, image.shape)
                initial_population.append(noise)

        # attack 3: Square Attack
        try:
            x_batch = np.expand_dims(image, axis=0).astype(np.float32)
            sq = SquareAttack(
                estimator=classifier,
                norm=np.inf,
                eps=self.config.epsilon,
                max_iter=1301,
                batch_size=1,
                verbose=False
            )
            adv_sq = sq.generate(x=x_batch)
            if adv_sq is None:
                adv_img = image
            else:
                adv_img = adv_sq[0]

            for _ in range(q):
                noise  = adv_img - image
                noise += np.random.normal(0, self.config.epsilon / 4, size=noise.shape)
                initial_population.append(self._clip_perturbation(noise))
        except Exception as e:
            warnings.warn(f"Square attack failed (falling back to random): {e}")
            for _ in range(q):
                noise = np.random.uniform(-self.config.epsilon, self.config.epsilon, image.shape)
                initial_population.append(noise)

        return initial_population

    def run_attack(self,
                         x_batch: np.ndarray,
                         y_batch: np.ndarray):
        """
        Run a mixed-method attack:
          1. Square Attack on target model to get black-box seed & query count.
          2. Use surrogate model to build initial EA population (Boundary + PGD + Square).
          3. Run CMA-ES on that population for the target model.

        Returns: (total_evo_queries, total_square_queries, evo_successes, square_successes).
        """
        if self.surrogate_model is None:
            raise ValueError("Surrogate model required for mixed method attack")

        # ensure correct dtype/device
        self.target_model    = self.target_model.float().to(self.config.device)
        self.surrogate_model = self.surrogate_model.float().to(self.config.device)

        if not os.path.exists('./results'):
            os.makedirs('./results')

        total_evo_q = total_sq_q = evo_ex = sq_ex = 0
        results_file = f"./results/results_{self.config.dataset}.txt"

        for i, (x_orig, y_true) in enumerate(zip(x_batch, y_batch)):
            if self.config.verbose:
                print(f"Processing sample {i+1}/{len(x_batch)}")

            # cnvert tensor -> numpy array of shape (C, H, W)
            if isinstance(x_orig, torch.Tensor):
                x_np = x_orig.detach().cpu().numpy().astype(np.float32)
            else:
                x_np = x_orig.astype(np.float32)

            # ensure exactly (C, H, W)
            if x_np.ndim == 3:
                base_img = x_np
            elif x_np.ndim == 4 and x_np.shape[0] == 1:
                base_img = np.squeeze(x_np, axis=0)
            else:
                raise ValueError(f"Unexpected input shape {x_np.shape} for mixed attack")

            # Square Attack queries (on target)
            target_clf = self._create_art_classifier(self.target_model)
            sq_queries = self._run_square_attack(target_clf, base_img)

            # 2. Build initial population using surrogate
            sur_clf = self._create_art_classifier(self.surrogate_model)
    
            init_noises = self._generate_initial_population(base_img, sur_clf)
            init_pop = np.stack(init_noises, axis=0).astype(np.float32)

            # 3. run CMA-ES refinement
            cma_att = CMAESAttack(self.target_model, self.surrogate_model, self.config)

            adv_img, _, evo_iters, _ = cma_att._run_cmaes(
                x_original=base_img,
                true_label=int(y_true),
                initial_perturbations=init_pop.reshape(init_pop.shape[0], -1),
                target_mode=True
            )

            # save comparison (original vs. adversarial numpy arrays)
            save_comparison_image(base_img, adv_img, f"{self.config.dataset}_{i}", self.config.dataset)

            # update statistics
            if evo_iters is False:
                # treat as no success (or use max possible)
                evo_iters = self.config.max_iter * self.config.population_size

            if evo_iters <= 1000:
                total_evo_q += evo_iters
                evo_ex      += 1

            if sq_queries <= 1000:
                total_sq_q += sq_queries
                sq_ex      += 1

            if self.config.verbose:
                print(f"Sample {i}: Evo {evo_iters} Square {sq_queries}")

            # append to results file
            with open(results_file, "a") as f:
                f.write(f"<{evo_iters}, {sq_queries}>\n")

        if self.config.verbose and evo_ex > 0:
            avg_evo = total_evo_q / evo_ex
            avg_sq = total_sq_q / sq_ex if sq_ex > 0 else 0
            print(f"\nAverage queries - Evolutionary: {avg_evo:.2f}, Square: {avg_sq:.2f}")

        return total_evo_q, total_sq_q, evo_ex, sq_ex
