from typing import Optional, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from utils.visualization import save_comparison_image
from attacks.config import AdversarialAttackConfig

class EvolutionaryAttack:
    """
    Base class for evolutionary adversarial attacks (CMA-ES, GA, mixed).
    """

    def __init__(self,
                 target_model: nn.Module,
                 surrogate_model: Optional[nn.Module] = None,
                 config: Optional[AdversarialAttackConfig] = None):
        self.target_model    = target_model
        self.surrogate_model = surrogate_model
        self.config          = config or AdversarialAttackConfig()

        # attack statistics
        self.queries        = 0
        self.target_queries = 0

        self.bonus = 999_999_999
        self.found = False

        # models in eval() mode
        self.target_model.eval()
        if self.surrogate_model is not None:
            self.surrogate_model.eval()


    def _model_predict(self, model: nn.Module, x: np.ndarray) -> np.ndarray:
        """
        Do a forward pass on `model` with numpy array `x`.
        Increments self.queries by 1 each time it's called.
        """
        self.queries += 1
        with torch.no_grad():
            tx  = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.config.device)
            out = model(tx)
        return out.cpu().numpy()


    def _get_prediction_class(self, model: nn.Module, x: np.ndarray) -> int:
        """
        Return the argmax class index from model(x) without incrementing queries permanently.
        """
        pred = self._model_predict(model, x)
        self.queries -= 1  # undo double count
        return int(np.argmax(pred))


    def _project_l2_ball(self, perturbation: np.ndarray, epsilon: float) -> np.ndarray:
        """Project `perturbation` onto an L2 ball of radius `epsilon`."""
        norm = np.linalg.norm(perturbation)
        if norm > epsilon: perturbation = perturbation * (epsilon / norm)
        return perturbation


    def _clip_perturbation(self, perturbation: np.ndarray) -> np.ndarray:
        """
        Clip each perturbation value into the interval [-ε, +ε].
        """
        return np.clip(perturbation, -self.config.epsilon, self.config.epsilon)


    def _save_comparison_image(self,
                               original: np.ndarray,
                               adversarial: np.ndarray,
                               tag: str = "") -> None:
        """
        Save side-by-side comparison. Delegates to utils.visualization.
        """
        save_comparison_image(original, adversarial, tag, self.config.dataset)
