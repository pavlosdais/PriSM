import os
import sys
import argparse
import torch
import numpy as np
from torchvision import models

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)

from utils.data_loader import import_dataset
from utils.attack_runner import test_attack
from utils.evaluation import calc_centroid, test_accuracy, get_latent_embedding, get_latent_hook

# import model classes
from models.mnist_models.models import Model_A, Model_B, Surrogate_Model
from models.cifar10_models.vgg import vgg16_bn
from models.cifar10_models.resnet import resnet18
from models.cifar10_models.inception import inception_v3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run adversarial attacks with configurable dataset, model, and attack."
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["mnist", "cifar10", "imagenet"],
        required=True,
        help="Which dataset to use."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        choices=[
            "model_a", "model_b", "inception", "vgg", "widenet", 
            "wideresnet", "efficientnet_b1", "densenet121",
            "convnext_tiny_robust", "vit_small_robust"
        ],
        help="Model to load."
    )
    parser.add_argument(
        "--attack", "-a",
        type=str,
        required=True,
        choices=["TASI", "SEGI", "SGSA"],
        help="Attack type."
    )
    parser.add_argument(
        "--epsilon", "-e",
        type=float,
        required=True,
        help="Epsilon value for the attack."
    )
    parser.add_argument(
        "--batch-start", "-b",
        type=int,
        default=0,
        help="Number of initial batches to skip."
    )
    parser.add_argument(
        "--calc-acc",
        action="store_true",
        help="If set, evaluate clean model accuracy before running attack."
    )
    parser.add_argument(
        "--device", 
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use: 'auto' (default), 'cpu', or 'cuda'."
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID to use if CUDA is selected (default: 0)."
    )
    
    return parser.parse_args()

def setup_device(device_arg, gpu_id):
    """Sets up and returns the appropriate torch device."""

    if device_arg == "auto":
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    elif device_arg == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else: device = torch.device(f"cuda:{gpu_id}")
    else: # cpu
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    return device

def load_data_with_device(dataset_name, seed, batch_size, device):
    """Loads dataset and creates data loaders with proper device handling."""

    num_classes, in_channels, train_data, test_data = import_dataset(dataset_name, seed)
    if train_data is None: sys.exit(1)
    
    pin_memory = device.type == 'cuda'
    num_workers = 4 if device.type == 'cuda' else 1
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return num_classes, in_channels, train_loader, test_loader

if __name__ == "__main__":
    args = parse_args()

    # setup device
    device = setup_device(args.device, args.gpu_id)

    # attack parameters
    dataset_name  = args.dataset
    model_choice  = args.model.lower()
    attack_choice = args.attack.lower()
    batch_start   = args.batch_start
    calc_acc      = args.calc_acc
    epsilon       = args.epsilon
    
    batch_size = 16 if dataset_name == "imagenet" else 32
    surr_model = None

    # set seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed)

    # step 1: load dataset
    try:
        num_classes, in_channels, train_loader, test_loader = load_data_with_device(dataset_name, seed, batch_size, device)
    except Exception as e:
        print(f"[ERROR] loading {dataset_name}: {e}")
        sys.exit(1)

    # step 2: instantiate models
    try:
        # mnist dataset
        if dataset_name == "mnist":
            if model_choice == "model_a":
                model        = Model_A(in_channels, num_classes)
                weights_file = "./models/mnist_models/target_model_a.pth"
            elif model_choice == "model_b":
                model        = Model_B(in_channels, num_classes)
                weights_file = "./models/mnist_models/target_model_b.pth"
            else:
                raise ValueError(f"Unknown MNIST model '{model_choice}'")

            if not os.path.exists(weights_file):
                raise FileNotFoundError(f"MNIST weights not found: {weights_file}")
            model.load_state_dict(torch.load(weights_file, map_location="cpu"))

            surr_model = Surrogate_Model()
            surr_file = "/kaggle/input/tgea-thesis/surrogate_model.pth"
            if os.path.exists(surr_file):
                surr_model.load_state_dict(torch.load(surr_file, map_location="cpu"))

        # cifar10 dataset
        elif dataset_name == "cifar10":
            if model_choice == "inception": model = inception_v3(pretrained=True)
            elif model_choice == "vgg":     model = vgg16_bn(pretrained=True)
            elif model_choice == "widenet":
                from robustbench.utils import load_model
                model = load_model(
                    model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf'
                )
            elif model_choice == "wideresnet":
                from robustbench.utils import load_model
                model = load_model(
                    model_name='Bartoldson2024Adversarial_WRN-94-16', dataset='cifar10', threat_model='Linf'
                )
            else: raise ValueError(f"Unknown CIFAR-10 model '{model_choice}'")

            surr_model = resnet18(pretrained=True)
            hook = surr_model.avgpool.register_forward_hook(get_latent_hook)
            surr_model.get_latent_embedding = get_latent_embedding.__get__(surr_model)

        # imagenet dataset
        elif dataset_name == "imagenet":
            if model_choice == "efficientnet_b1":
                model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
            elif model_choice == "densenet121":
                model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            else:
                raise ValueError(f"Unknown or unsupported ImageNet model '{model_choice}'")

            surr_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            hook       = surr_model.avgpool.register_forward_hook(get_latent_hook)
            surr_model.get_latent_embedding = get_latent_embedding.__get__(surr_model)
        
        else:
            raise ValueError(f"Unsupported dataset '{dataset_name}'")

        # move models to the selected device
        model.to(device)
        if surr_model: surr_model.to(device)
        print(f"Models loaded and moved to {device}")

    except Exception as e:
        print(f"[ERROR] loading model(s): {e}")
        sys.exit(1)

    # step 3: (optionally) calculate accuracy
    if calc_acc:
        try:
            acc = test_accuracy(model, test_loader, device=device)
            print(f"Target Model Accuracy: {acc:.2f}%")
            if surr_model:
                sacc = test_accuracy(surr_model, test_loader, device=device)
                print(f"Surrogate Model Accuracy: {sacc:.2f}%")
        except Exception as e:
            print(f"[ERROR] during accuracy test: {e}")

    # step 4: run adversarial attack
    print(f"\n>> Running {attack_choice.upper()} attack on {model_choice} ({dataset_name.upper()})")
    try:
        centroids = calc_centroid(surr_model, train_loader, num_classes=num_classes, device=device) if surr_model and attack_choice == 'segi' else None
        total = test_attack(
            model=model,
            surr_model=surr_model,
            loader=test_loader,
            class_centroids=centroids,
            dataset_name=dataset_name,
            batches_skip=batch_start,
            attack_type=attack_choice,
            verbose=True,
            epsilon_v=epsilon,
            device=device.type
        )
        print(f"\n>> Completed. Processed {total} samples.")
    except KeyboardInterrupt:
        print("\n>> Attack interrupted by user.")
    except Exception as e:
        print(f"[ERROR] during attack run: {e}")
        import traceback; traceback.print_exc()
    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
