import os
import numpy as np
import matplotlib.pyplot as plt


def save_comparison_image(original: np.ndarray, adversarial: np.ndarray, tag: str = "", dataset: str = "mnist"):
    """
    Save side-by-side comparison of original vs. adversarial images under ./img/.
    """
    if not os.path.exists('./img'):
        os.makedirs('./img')

    if dataset.lower() == 'mnist':
        orig_img = np.squeeze(np.clip(original, 0, 1))
        adv_img  = np.squeeze(np.clip(adversarial, 0, 1))
        cmap     = 'gray'
    else:
        orig_img = np.transpose(np.clip(original, 0, 1), (1, 2, 0))
        adv_img  = np.transpose(np.clip(adversarial, 0, 1), (1, 2, 0))
        cmap     = None

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(orig_img, cmap=cmap)
    ax1.set_title("Original")
    ax1.axis('off')

    ax2.imshow(adv_img, cmap=cmap)
    ax2.set_title("Adversarial")
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(f'./img/comparison_{tag}.png', dpi=150, bbox_inches='tight')
    plt.close()
