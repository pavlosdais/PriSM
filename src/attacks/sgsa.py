from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from tqdm.auto import trange

from art.attacks.evasion import SquareAttack
from art.utils import check_and_transform_label_format, get_labels_np_array
from art.config import ART_NUMPY_DTYPE

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)

class SGSquareAttack(SquareAttack):
    attack_params = SquareAttack.attack_params + [
        "surrogate_estimator", "saliency_scale_factor", "gaussian_sigma", 
        "saliency_update_frequency", "fallback_threshold", "random_fallback_prob",
        "save_path", "attention_estimator", "scales", "temporal_decay", "attention_weight"
    ]

    def __init__(
        self,
        estimator: "CLASSIFIER_TYPE",
        surrogate_estimator: "CLASSIFIER_TYPE",
        attention_estimator: "CLASSIFIER_TYPE" = None,
        saliency_scale_factor: float = 10.0,
        gaussian_sigma: float = 1.0,
        saliency_update_frequency: int = 50,
        fallback_threshold: int = 100,
        random_fallback_prob: float = 0.05,
        save_path: str = "results",
        scales: list = [1.0, 0.5, 0.25],
        temporal_decay: float = 0.9,
        attention_weight: float = 0.3,
        **kwargs,
    ):
        self.surrogate_estimator = surrogate_estimator
        self.attention_estimator = attention_estimator
        self.saliency_scale_factor = saliency_scale_factor
        self.gaussian_sigma = gaussian_sigma
        self.saliency_update_frequency = saliency_update_frequency
        self.fallback_threshold = fallback_threshold
        self.random_fallback_prob = random_fallback_prob
        self.save_path = save_path
        self.scales = scales
        self.temporal_decay = temporal_decay
        self.attention_weight = attention_weight
        
        kwargs["norm"] = np.inf
        super().__init__(estimator=estimator, **kwargs)

        if not hasattr(surrogate_estimator, "loss_gradient"):
            raise ValueError("The surrogate estimator must have a 'loss_gradient' method.")
    
    def _save_visualizations(self, x: np.ndarray, x_adv: np.ndarray, saliency_map: np.ndarray):
        logger.info(f"Saving {len(x_adv)} adversarial examples and visualizations to {self.save_path}")
        os.makedirs(self.save_path, exist_ok=True)
        
        import matplotlib.pyplot as plt
        for i, (orig_img, adv_img, sal_map) in enumerate(zip(x, x_adv, saliency_map)):
            if self.estimator.channels_first:
                orig_img_plt = np.transpose(orig_img, (1, 2, 0))
                adv_img_plt = np.transpose(adv_img, (1, 2, 0))
            else:
                orig_img_plt = orig_img
                adv_img_plt = adv_img

            orig_img_plt = np.clip(orig_img_plt, 0, 1)
            adv_img_plt = np.clip(adv_img_plt, 0, 1)

            if orig_img_plt.shape[-1] == 1:
                orig_img_plt = orig_img_plt.squeeze(axis=-1)
                adv_img_plt = adv_img_plt.squeeze(axis=-1)
                cmap_img = 'gray'
            else:
                cmap_img = None

            fig, axes = plt.subplots(1, 3, figsize=(24, 8))

            axes[0].imshow(orig_img_plt, cmap=cmap_img)
            axes[0].set_title("Original Image", fontsize=20)
            axes[0].axis('off')

            sal_map_norm = (sal_map - np.min(sal_map)) / (np.max(sal_map) - np.min(sal_map) + 1e-9)
            axes[1].imshow(sal_map_norm, cmap='hot')
            axes[1].set_title("Multi-scale Saliency Map", fontsize=20)
            axes[1].axis('off')

            axes[2].imshow(adv_img_plt, cmap=cmap_img)
            axes[2].set_title("Adversarial Image", fontsize=20)
            axes[2].axis('off')
            
            plt.tight_layout()
            fig.savefig(os.path.join(self.save_path, f"comparison_{i:04d}.png"), dpi=200)
            plt.close(fig)

    def _compute_multiscale_saliency(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes a multi-scale saliency map for a batch of input images.
        
        This method generates a vulnerability heatmap that is more robust than a
        standard saliency map by considering features at multiple resolutions. It
        first computes a base saliency map from the surrogate model's gradients.
        It then creates several versions of this map at different scales (e.g.,
        1.0x, 0.5x, 0.25x), optionally smooths them with a Gaussian filter, and
        finally averages them together. This process captures both fine-grained
        details and broader, lower-frequency vulnerable regions, providing a more
        comprehensive guide for the attack.
        """
    
        grads = self.surrogate_estimator.loss_gradient(x, y)
        if self.estimator.channels_first:
            base_saliency = np.sum(np.abs(grads), axis=1)
        else:
            base_saliency = np.sum(np.abs(grads), axis=-1)
        
        multiscale_saliency = np.zeros_like(base_saliency)
        
        for scale in self.scales:
            if scale == 1.0:
                scaled_saliency = base_saliency
            else:
                scaled_saliency = np.array([
                    zoom(s, scale, order=1) for s in base_saliency
                ])
                scaled_saliency = np.array([
                    zoom(s, 1.0/scale, order=1) for s in scaled_saliency
                ])
                
            if self.gaussian_sigma > 0:
                scaled_saliency = np.array([
                    gaussian_filter(s, sigma=self.gaussian_sigma * scale) 
                    for s in scaled_saliency
                ])
            
            multiscale_saliency += scaled_saliency / len(self.scales)
        
        return multiscale_saliency

    def _compute_attention_map(self, x: np.ndarray) -> np.ndarray:
        """
        Computes an attention map from the attention_estimator.

        Computes an attention map from the attention_estimator.
        This map provides a complementary signal to the saliency map, highlighting
        regions the model focuses on for its prediction rather than just regions
        with high gradient magnitudes. If the estimator has a dedicated get_attention
        method, it is used; otherwise, a standard gradient-based heatmap is computed.
        """

        if self.attention_estimator is None:
            return np.zeros((x.shape[0], x.shape[-2], x.shape[-1]))
        
        try:
            if hasattr(self.attention_estimator, 'get_attention'):
                attention_maps = self.attention_estimator.get_attention(x)
                if attention_maps.ndim == 4:
                    attention_maps = np.mean(attention_maps, axis=1)
                return attention_maps
            else:
                predictions  = self.attention_estimator.predict(x)
                pred_indices = np.argmax(predictions, axis=1)
                grads        = self.attention_estimator.loss_gradient(x, pred_indices)
                
                if self.estimator.channels_first:
                    attention_maps = np.sum(np.abs(grads), axis=1)
                else:
                    attention_maps = np.sum(np.abs(grads), axis=-1)
                
                return attention_maps
        except:
            return np.zeros((x.shape[0], x.shape[-2], x.shape[-1]))

    def _get_temporal_update_schedule(self, iteration: int, max_iter: int) -> bool:
        """
        Determines if the saliency map should be updated at the current iteration.

        This function implements a dynamic schedule where saliency maps are re-computed
        more frequently at the beginning of the attack and less frequently as the
        attack progresses. This balances the need for up-to-date guidance with
        the computational cost of generating new maps
        """

        base_freq = self.saliency_update_frequency
        
        if iteration < max_iter * 0.1:
            return iteration % (base_freq // 4) == 0
        elif iteration < max_iter * 0.5:
            return iteration % (base_freq // 2) == 0
        else:
            return iteration % base_freq == 0

    def _should_use_fallback(self, iterations_without_improvement: np.ndarray, current_iter: int) -> np.ndarray:
        """
        Decides whether to revert to the baseline random search for a step.

        The fallback mechanism ensures robustness. It is triggered under two conditions:
        1.  Stagnation-Based: If the attack has failed to improve the loss for
            a certain number of consecutive iterations (defined by `fallback_threshold`).
        2.  Stochastic: A small, random probability is used at each step to
            encourage exploration and prevent the attack from getting stuck in a
            local optimum of the saliency map.
        """

        if np.any(iterations_without_improvement >= self.fallback_threshold): return True
        return np.random.random(len(iterations_without_improvement)) < self.random_fallback_prob

    def _apply_random_attack(self, x_robust: np.ndarray, x_init: np.ndarray, 
                           height: int, width: int, channels: int, 
                           percentage_of_elements: float) -> np.ndarray:
        x_new_batch = x_robust.copy()
        base_height_tile = max(int(round(math.sqrt(percentage_of_elements * height * width))), 1)
        
        for i in range(len(x_new_batch)):
            height_tile = width_tile = base_height_tile
            h_start = np.random.randint(0, height - height_tile + 1)
            w_start = np.random.randint(0, width - width_tile + 1)
            
            delta_val = np.random.choice([-1, 1]) * 2 * self.eps
            
            if self.estimator.channels_first:
                x_new_batch[i, :, h_start:h_start+height_tile, w_start:w_start+width_tile] += delta_val
            else:
                x_new_batch[i, h_start:h_start+height_tile, w_start:w_start+width_tile, :] += delta_val
        
        x_new_batch = np.minimum(np.maximum(x_new_batch, x_init - self.eps), x_init + self.eps)
        x_new_batch = np.clip(x_new_batch, self.estimator.clip_values[0], self.estimator.clip_values[1])
        
        return x_new_batch

    def _apply_attention_guided_attack(self, x_robust: np.ndarray, x_init: np.ndarray, 
                                     combined_saliency: np.ndarray, height: int, width: int, 
                                     channels: int, percentage_of_elements: float) -> np.ndarray:
        x_new_batch = x_robust.copy()
        base_height_tile = max(int(round(math.sqrt(percentage_of_elements * height * width))), 1)
        
        # convert the saliecy map into a probability distribution
        # "brighter" spots are more important
        saliency_flat = combined_saliency.reshape(combined_saliency.shape[0], -1)
        saliency_flat = np.maximum(saliency_flat, 1e-10)
        saliency_flat_sum = np.sum(saliency_flat, axis=1, keepdims=True)
        prob_dist = saliency_flat / saliency_flat_sum
        prob_dist = prob_dist / np.sum(prob_dist, axis=1, keepdims=True)
        
        # pick a pixel from the distribution
        flat_idx = np.array([np.random.choice(saliency_flat.shape[1], p=p) for p in prob_dist])
        row_idx, col_idx = np.unravel_index(flat_idx, (height, width))
        
        # the size of the square depends on how "hot"/probable/weak the chosen pixel is
        local_saliency = np.array([s[r, c] for s, r, c in zip(combined_saliency, row_idx, col_idx)])
        saliency_factor = 1.0 / (1.0 + self.saliency_scale_factor * local_saliency)

        tile_size = np.clip(np.round(base_height_tile * saliency_factor), 1, min(height, width)).astype(int)
        height_tile = width_tile = tile_size
        
        # apply the perturbation
        h_start = np.array([np.clip(r - ht // 2, 0, height - ht) for r, ht in zip(row_idx, height_tile)])
        w_start = np.array([np.clip(c - ht // 2, 0, width - ht) for c, ht in zip(col_idx, height_tile)])
        
        for i in range(len(x_new_batch)):
            delta_val = np.random.choice([-1, 1]) * 2 * self.eps
            if self.estimator.channels_first:
                x_new_batch[i, :, h_start[i]:h_start[i]+height_tile[i], w_start[i]:w_start[i]+height_tile[i]] += delta_val
            else:
                x_new_batch[i, h_start[i]:h_start[i]+height_tile[i], w_start[i]:w_start[i]+height_tile[i], :] += delta_val
        
        x_new_batch = np.minimum(np.maximum(x_new_batch, x_init - self.eps), x_init + self.eps)
        x_new_batch = np.clip(x_new_batch, self.estimator.clip_values[0], self.estimator.clip_values[1])
        
        return x_new_batch

    def _update_temporal_saliency(self, x_current: np.ndarray, y_current: np.ndarray, 
                                previous_saliency: np.ndarray, attention_map: np.ndarray) -> np.ndarray:
        new_multiscale_saliency = self._compute_multiscale_saliency(x_current, y_current)
        """
        Updates the guidance map using temporal smoothing.

        This function creates the final, comprehensive guidance map for the current
        step. It first computes a new multi-scale saliency map, combines it with the
        attention map, and then uses an exponential moving average to smooth this
        result with the guidance map from the previous update. This makes the
        guidance more stable and adaptive over time.
        """
        
        combined_saliency = (1 - self.attention_weight) * new_multiscale_saliency + \
                          self.attention_weight * attention_map
        
        temporal_saliency = self.temporal_decay * previous_saliency + \
                           (1 - self.temporal_decay) * combined_saliency
        
        return temporal_saliency

    def generate(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> np.ndarray:
        if self.norm not in [np.inf, "inf"]:
            raise ValueError("This attack is designed for the l-infinity norm.")

        x_adv = x.astype(ART_NUMPY_DTYPE)

        if y is None:
            logger.info("Using model predictions as true labels.")
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        
        y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        multiscale_saliency = self._compute_multiscale_saliency(x, y)
        attention_map       = self._compute_attention_map(x)
        combined_saliency   = (1 - self.attention_weight) * multiscale_saliency + self.attention_weight * attention_map

        if self.estimator.channels_first:
            channels, height, width = x.shape[1], x.shape[2], x.shape[3]
        else:
            height, width, channels = x.shape[1], x.shape[2], x.shape[3]

        for _ in trange(self.nb_restarts, desc="Saliency Guided SquareAttack - restarts", disable=not self.verbose):
            y_pred = self.estimator.predict(x_adv, batch_size=self.batch_size)
            sample_is_robust = np.logical_not(self.adv_criterion(y_pred, y))

            if np.sum(sample_is_robust) == 0:
                break
            
            x_robust = x[sample_is_robust]
            y_robust = y[sample_is_robust]
            saliency_robust = combined_saliency[sample_is_robust]
            
            x_adv_best_in_restart = x_robust.copy()
            sample_loss_init = self.loss(x_adv_best_in_restart, y_robust)
            
            iterations_without_improvement = np.zeros(len(x_robust), dtype=int)

            if self.estimator.channels_first:
                size = (x_robust.shape[0], channels, 1, width)
            else:
                size = (x_robust.shape[0], 1, width, channels)
            
            x_robust_new = np.clip(
                x_robust + self.eps * np.random.choice([-1, 1], size=size),
                a_min=self.estimator.clip_values[0],
                a_max=self.estimator.clip_values[1],
            ).astype(ART_NUMPY_DTYPE)

            sample_loss_new = self.loss(x_robust_new, y_robust)
            loss_improved = (sample_loss_new - sample_loss_init) < 0.0
            
            x_adv_best_in_restart[loss_improved] = x_robust_new[loss_improved]
            sample_loss_init[loss_improved] = sample_loss_new[loss_improved]

            for i_iter in trange(
                    self.max_iter, desc="Saliency Guided SquareAttack - iterations", leave=False, disable=not self.verbose
                ):
                
                percentage_of_elements = self._get_percentage_of_elements(i_iter)
                
                # Determine correctly predicted samples
                y_pred = self.estimator.predict(x_adv_best_in_restart, batch_size=self.batch_size)
                iter_is_robust = np.logical_not(self.adv_criterion(y_pred, y_robust))

                if np.sum(iter_is_robust) == 0:
                    break
                
                # Update saliecy map as the attack progresses
                if i_iter > 0 and self._get_temporal_update_schedule(i_iter, self.max_iter):
                    attention_map_current = self._compute_attention_map(x_adv_best_in_restart)
                    combined_saliency = self._update_temporal_saliency(
                        x_adv_best_in_restart, y_robust, combined_saliency, attention_map_current
                    )
                    saliency_robust = combined_saliency[sample_is_robust]

                x_robust_iter = x_adv_best_in_restart[iter_is_robust]
                x_init_iter = x_robust[iter_is_robust]
                y_robust_iter = y_robust[iter_is_robust]
                saliency_iter = saliency_robust[iter_is_robust]
                loss_iter = sample_loss_init[iter_is_robust]
                iters_without_improvement = iterations_without_improvement[iter_is_robust]

                use_fallback = self._should_use_fallback(iters_without_improvement, i_iter)
                
                # Use random attack in case of fallback
                if np.any(use_fallback):
                    fallback_indices = np.where(use_fallback)[0]
                    x_new_batch = x_robust_iter.copy()
                    if len(fallback_indices) > 0:
                        x_fallback = self._apply_random_attack(
                            x_robust_iter[fallback_indices],
                            x_init_iter[fallback_indices],
                            height, width, channels,
                            percentage_of_elements
                        )
                        x_new_batch[fallback_indices] = x_fallback
                    
                    saliency_indices = np.where(~use_fallback)[0]
                    if len(saliency_indices) > 0:
                        x_saliency = self._apply_attention_guided_attack(
                            x_robust_iter[saliency_indices],
                            x_init_iter[saliency_indices],
                            saliency_iter[saliency_indices],
                            height, width, channels,
                            percentage_of_elements
                        )
                        x_new_batch[saliency_indices] = x_saliency
                else:
                    x_new_batch = self._apply_attention_guided_attack(
                        x_robust_iter, x_init_iter, saliency_iter,
                        height, width, channels, percentage_of_elements
                    )
                
                sample_loss_new = self.loss(x_new_batch, y_robust_iter)
                loss_improved = (sample_loss_new - loss_iter) < 0.0
                
                iters_without_improvement[~loss_improved] += 1
                iters_without_improvement[loss_improved] = 0
                
                x_robust_iter[loss_improved] = x_new_batch[loss_improved]
                loss_iter[loss_improved] = sample_loss_new[loss_improved]
                
                x_adv_best_in_restart[iter_is_robust] = x_robust_iter
                sample_loss_init[iter_is_robust] = loss_iter
                iterations_without_improvement[iter_is_robust] = iters_without_improvement
            
            x_adv[sample_is_robust] = x_adv_best_in_restart

        # Save image
        if self.save_path is not None:
            self._save_visualizations(x, x_adv, combined_saliency)
        
        return x_adv

    def _check_params(self) -> None:
        super()._check_params()
        
        if hasattr(self, 'saliency_scale_factor') and self.saliency_scale_factor <= 0:
            raise ValueError("The saliency scale factor must be positive.")
        if hasattr(self, 'gaussian_sigma') and self.gaussian_sigma < 0:
            raise ValueError("The Gaussian sigma must be non-negative.")
        if hasattr(self, 'saliency_update_frequency') and self.saliency_update_frequency <= 0:
            raise ValueError("The saliency update frequency must be positive.")
        if hasattr(self, 'fallback_threshold') and self.fallback_threshold <= 0:
            raise ValueError("The fallback threshold must be positive.")
        if hasattr(self, 'random_fallback_prob') and not (0 <= self.random_fallback_prob <= 1):
            raise ValueError("The random fallback probability must be between 0 and 1.")
        if hasattr(self, 'temporal_decay') and not (0 <= self.temporal_decay <= 1):
            raise ValueError("The temporal decay must be between 0 and 1.")
        if hasattr(self, 'attention_weight') and not (0 <= self.attention_weight <= 1):
            raise ValueError("The attention weight must be between 0 and 1.")
        if hasattr(self, 'scales') and not all(s > 0 for s in self.scales):
            raise ValueError("All scales must be positive.")
