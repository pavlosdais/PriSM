import os
import numpy as np
import torch
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import SimBA

from .sgsa import SGSquareAttack

class SGSQInitialization():
    def __init__(self, target_model, surrogate_model, config):
        self.target_model = target_model
        self.surrogate_model = surrogate_model
        self.config = config
        
        # ensure device is properly set
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    def _create_art_classifier(self, model, for_simba=False):
        """
        Wrap a PyTorch nn.Module into ART's PyTorchClassifier to run attacks.
        """

        # move model to the correct device
        model = model.to(self.device)
        
        dummy_input = torch.zeros(1, *self.config.input_shape, dtype=torch.float32).to(self.device)
        model.eval()
        with torch.no_grad():    dummy_out = model(dummy_input)
        if dummy_out.dim() == 1: dummy_out = dummy_out.unsqueeze(0)
        if dummy_out.dim() > 2:  dummy_out = dummy_out.view(dummy_out.shape[0], -1)
        inferred_nb_classes = dummy_out.shape[-1]

        if for_simba:
            class ProbabilityWrapper(torch.nn.Module):
                def __init__(self, model, device):
                    super().__init__()
                    self.model = model
                    self.device = device
                
                def forward(self, x):
                    # Ensure input is on the correct device
                    if isinstance(x, torch.Tensor):
                        x = x.to(self.device)
                    elif isinstance(x, np.ndarray):
                        x = torch.from_numpy(x).to(self.device)
                    
                    with torch.no_grad():
                        if self.model.training:
                            self.model.eval()
                        
                        logits = self.model(x)
                        
                        if logits.dim() > 2:
                            logits = logits.view(logits.shape[0], -1)
                        if logits.dim() == 1:
                            logits = logits.unsqueeze(0)
                        
                        if logits.shape[-1] == 1:
                            probs = torch.sigmoid(logits)
                            return torch.cat([1 - probs, probs], dim=-1)
                        else:
                            return torch.nn.functional.softmax(logits, dim=-1)
            
            wrapped_model = ProbabilityWrapper(model, self.device)
        else:
            wrapped_model = model
            
        return PyTorchClassifier(
            model=wrapped_model,
            clip_values=(0, 1),
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(wrapped_model.parameters(), lr=0.01),
            input_shape=self.config.input_shape,
            nb_classes=inferred_nb_classes,
            device_type='gpu' if self.device.type == 'cuda' else 'cpu'
        )

    def _run_attack_and_count_queries(self, attack, x_in: np.ndarray, y_in: np.ndarray | None, attack_name: str) -> tuple[int, bool]:
        """
        A helper function to run an attack and count the queries to the underlying model.
        Returns (query_count, success)
        """

        query_count = 0
        
        # handle different attack interfaces
        target_estimator = attack.estimator if hasattr(attack, 'estimator') else attack.classifier
        
        orig_predict = target_estimator.predict

        def counting_predict(x, batch_size=1):
            nonlocal query_count
            query_count += x.shape[0]
            
            # ensure input is float32 and handle device placement
            if isinstance(x, np.ndarray): x = x.astype(np.float32)
            
            result = orig_predict(x, batch_size=batch_size)
            
            if result.ndim == 1:
                result = result.reshape(1, -1)
            
            return result

        target_estimator.predict = counting_predict
        
        # prepare target labels for ART format
        y_art = None
        if y_in is not None:
            num_classes = target_estimator.nb_classes
            y_art = np.eye(num_classes)[y_in.astype(int)]
            if y_art.ndim > 2:
                y_art = np.squeeze(y_art, axis=1)
        
        try:
            if x_in.ndim == 3:
                x_in = x_in.reshape(1, *x_in.shape)
            
            # ensure input is float32
            x_in = x_in.astype(np.float32)
            
            # generate the adversarial example
            adv_x = attack.generate(x=x_in, y=y_art)

            # check success by comparing predictions
            orig_pred = target_estimator.predict(x_in.astype(np.float32))
            adv_pred  = target_estimator.predict(adv_x.astype(np.float32))
            
            orig_class = np.argmax(orig_pred, axis=1).item()
            adv_class  = np.argmax(adv_pred, axis=1).item()

            success = orig_class != adv_class
            
        except Exception as e:
            print(f"Warning: {attack_name} failed with error: {e}")
            success = False

        # restore the original predict method to avoid nested counting
        target_estimator.predict = orig_predict
        
        # consider failed if too many queries
        if query_count > 1000: return query_count, False
        return query_count, success

    def run_attack(self, x_batch: np.ndarray, y_batch: np.ndarray, run_simba = False):
        """
        Runs the comparison between SimBA Attack and Saliency Guided Square Attack.
        """
        if self.surrogate_model is None:
            raise ValueError("A surrogate model is required for the Saliency Guided Square Attack.")

        # Move models to the correct device
        self.target_model    = self.target_model.float().to(self.device)
        self.surrogate_model = self.surrogate_model.float().to(self.device)

        if not os.path.exists('./results'):
            os.makedirs('./results')

        # adjust labels if they appear 1-based
        if self.config.dataset == "imagenet":
            y_batch = y_batch.astype(int)
            y_min = np.min(y_batch)
            y_max = np.max(y_batch)
            if y_min >= 1 and y_max == self.config.num_classes: y_batch -= 1

        total_simba_q = total_adv_sq_q = 0
        simba_ex = adv_sq_ex = 0
        results_file = f"./results/comparison_results_{self.config.dataset}.txt"

        target_clf = self._create_art_classifier(self.target_model)
        sur_clf    = self._create_art_classifier(self.surrogate_model)
        simba_clf  = self._create_art_classifier(self.target_model, for_simba=True)

        # instantiate both attacks once
        simba_attack = SimBA(
            classifier=simba_clf,
            attack='dct',
            max_iter=500,
            epsilon=self.config.epsilon,
            targeted=False,
            batch_size=1,
            verbose=False
        )

        sgsa_attack = SGSquareAttack(
            estimator=target_clf,
            surrogate_estimator=sur_clf,
            attention_estimator=sur_clf,
            eps=self.config.epsilon,
            max_iter=700,
            batch_size=1,
            verbose=False
        )

        for i, (x_orig, y_true) in enumerate(zip(x_batch, y_batch)):
            if self.config.verbose:
                print(f"Processing sample {i+1}/{len(x_batch)}")

            # Convert to numpy and ensure correct data type
            if isinstance(x_orig, torch.Tensor):
                x_np = x_orig.detach().cpu().numpy().astype(np.float32)
            else:
                x_np = x_orig.astype(np.float32)
                
            if isinstance(y_true, torch.Tensor): y_np = y_true.detach().cpu().numpy()
            else:                                y_np = np.array([y_true])
            
            # check for valid label
            try:               label = int(y_np[0])
            except ValueError: label = -1  # invalid conversion
                        
            if not 0 <= label < target_clf.nb_classes:
                if self.config.verbose:
                    print(f"Sample {i+1}: Invalid label {label}. Skipping.")
                with open(results_file, "a") as f:
                    f.write("<INVALID, INVALID>\n")
                continue
            
            x_in = np.expand_dims(x_np, axis=0)

            # step 1: run SimBA Attack
            if run_simba:
                simba_queries, simba_success = self._run_attack_and_count_queries(simba_attack, x_in, y_np, "SimBA")

            # step 2: run Saliency Guided Square Attack
            adv_sq_queries, adv_sq_success = self._run_attack_and_count_queries(sgsa_attack, x_in, y_np, "SG_Square")

            # update statistics
            if run_simba and simba_success:
                total_simba_q += simba_queries
                simba_ex += 1
            
            if adv_sq_success:
                total_adv_sq_q += adv_sq_queries
                adv_sq_ex += 1

            if self.config.verbose:
                adv_sq_status = "SUCCESS" if adv_sq_success else "FAILED"

                if run_simba:
                    simba_status = "SUCCESS" if simba_success else "FAILED"
                    print(f"Sample {i+1}: SimBA Queries={simba_queries} ({simba_status}), Saliency-Guided Square Attack Queries={adv_sq_queries} ({adv_sq_status})")
                else:
                    print(f"Sample {i+1}: Saliency-Guided Square Attack Queries={adv_sq_queries} ({adv_sq_status})")

            with open(results_file, "a") as f:
                if run_simba: f.write(f"<{simba_queries}, {adv_sq_queries}>\n")
                else:         f.write(f"<{adv_sq_queries}>\n")

        if self.config.verbose:
            avg_simba = total_simba_q / simba_ex if simba_ex > 0 else float('inf')
            avg_adv_sq = total_adv_sq_q / adv_sq_ex if adv_sq_ex > 0 else float('inf')
            print("\n" + "="*50)
            print("                Attack Comparison Results")
            print("="*50)
            print(f"Total Samples: {len(x_batch)}")
            print("\n--- SimBA Attack ---")
            print(f"Success Rate: {simba_ex / len(x_batch) * 100:.2f}% ({simba_ex}/{len(x_batch)})")
            print(f"Average Queries on Success: {avg_simba:.2f}")
            print("\n--- Advanced Square Attack (with Prior) ---")
            print(f"Success Rate: {adv_sq_ex / len(x_batch) * 100:.2f}% ({adv_sq_ex}/{len(x_batch)})")
            print(f"Average Queries on Success: {avg_adv_sq:.2f}")
            print("="*50)

        return total_adv_sq_q, total_simba_q, adv_sq_ex, simba_ex
