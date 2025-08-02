import os
import sys
import argparse
import torch
import numpy as np

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
        choices=["mnist", "cifar10"],
        required=True,
        help="Which dataset to use."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        choices=["model_a", "model_b", "inception", "vgg"],
        help="Model to load (e.g. 'model_a','model_b' for MNIST; 'inception','vgg' for CIFAR-10)."
    )
    parser.add_argument(
        "--attack", "-a",
        type=str,
        required=True,
        choices=["TASI", "SEGI", "SGSA"],
        help="Attack type: 'mixed', 'ga' (genetic), or 'cmaes'."
    )

    parser.add_argument(
        "--epsilon", "-e",
        type=float,
        required=True,
        help="Epsilon value."
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
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # attack parameters
    dataset_name  = args.dataset
    model_choice  = args.model.lower()
    attack_choice = args.attack.lower()
    batch_start   = args.batch_start
    calc_acc      = args.calc_acc
    epsilon       = args.epsilon
    batch_size    = 32
    surr_model    = None

    # fix random seeds
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # step 1: load dataset
    try:
        num_classes, in_channels, train_data, test_data = import_dataset(dataset_name, seed)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=False, num_workers=0
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=False, num_workers=0
        )
    except Exception as e:
        print(f"[ERROR] loading {dataset_name}: {e}")
        sys.exit(1)

    # step 2: instantiate models
    try:
        if dataset_name == "mnist":
            if model_choice == "model_a":
                model        = Model_A(in_channels, num_classes)
                weights_file = "./models/mnist_models/target_model_a.pth"
            elif model_choice == "model_b":
                model        = Model_B(in_channels, num_classes)
                weights_file = "./models/mnist_models/target_model_b.pth"
            else:
                raise ValueError(f"Unknown MNIST model '{model_choice}'")

            # load model weights
            if not os.path.exists(weights_file):
                raise FileNotFoundError(f"MNIST weights not found: {weights_file}")
            model.load_state_dict(torch.load(weights_file, map_location="cpu"))

            # surrogate
            surr_model = Surrogate_Model()
            surr_file = "./models/mnist_models/surrogate_model.pth"
            if os.path.exists(surr_file):
                surr_model.load_state_dict(torch.load(surr_file, map_location="cpu"))

        elif dataset_name == "cifar10":
            if model_choice == "inception":
                model = inception_v3(pretrained=True)
            elif model_choice == "vgg":
                model = vgg16_bn(pretrained=True)
            else:
                raise ValueError(f"Unknown CIFAR-10 model '{model_choice}'")

            # resnet18 surrogate
            surr_model = resnet18(pretrained=True)
            hook = surr_model.avgpool.register_forward_hook(get_latent_hook)
            surr_model.get_latent_embedding = get_latent_embedding.__get__(surr_model)

        else:
            raise ValueError(f"Unsupported dataset '{dataset_name}'")

    except Exception as e:
        print(f"[ERROR] loading model(s): {e}")
        sys.exit(1)

    # step 3: optional clean accuracy
    if calc_acc:
        try:
            acc = test_accuracy(model, test_loader, device="cpu")
            print(f"Target Model Accuracy: {acc:.2f}%")
            if surr_model:
                sacc = test_accuracy(surr_model, test_loader, device="cpu")
                print(f"Surrogate Model Accuracy: {sacc:.2f}%")
        except Exception as e:
            print(f"[ERROR] during accuracy test: {e}")

    # step 4: run adversarial attack
    print(f"\n>> Running {attack_choice.upper()} attack on {model_choice} ({dataset_name.upper()})")
    try:
        centroids = calc_centroid(surr_model, train_loader) if surr_model else None
        total = test_attack(
            model=model,
            surr_model=surr_model,
            loader=test_loader,
            class_centroids=centroids,
            dataset_name=dataset_name,
            batches_skip=batch_start,
            attack_type=attack_choice,
            verbose=True,
            epsilon_v=epsilon
        )
        print(f"\n>> Completed. Processed {total} samples.")
    except KeyboardInterrupt:
        print("\n>> Attack interrupted by user.")
    except Exception as e:
        print(f"[ERROR] during attack run: {e}")
        import traceback; traceback.print_exc()
