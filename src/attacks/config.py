class AdversarialAttackConfig:
    """Configuration class for adversarial attacks."""

    def __init__(self,
                 epsilon: float = 0.3,
                 max_iter: int = 80,
                 population_size: int = 250,
                 dataset: str = "mnist",
                 device: str = "cpu",
                 seed: int = 42,
                 verbose: bool = True,
                 centroids = None):
        
        # attack parameters
        self.epsilon         = epsilon
        self.max_iter        = max_iter
        self.population_size = population_size
        self.dataset         = dataset.lower()
        self.device          = device
        self.seed            = seed
        self.verbose         = verbose
        self.centroids       = centroids

        if self.dataset == "mnist":
            self.input_shape = (1, 28, 28)
            self.num_classes = 10
            self.class_names = [str(i) for i in range(10)]
        elif self.dataset == "cifar10":
            self.input_shape = (3, 32, 32)
            self.num_classes = 10
            self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                                'dog', 'frog', 'horse', 'ship', 'truck']
        elif self.dataset == "imagenet":
            self.input_shape = (3, 224, 224) 
            self.num_classes = 1000

            # massive list, skip it
            self.class_names = [] 
        else:
            raise ValueError(f"Dataset '{self.dataset}' is unsupported")
