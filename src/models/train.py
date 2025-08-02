import torch
import numpy as np

import time
import sys
from sklearn.metrics import f1_score, confusion_matrix

class Train():
    """
    A class to handle the training of a model.

    Parameters:
        model (nn.Module)        -- the neural network model to be trained
        device (torch.device)    -- the device on which to train the model (CPU or GPU)
        num_epochs (int)         -- the number of epochs to train the model
        optimizer (Optimizer)    -- the optimizer to use for training
        train_loader (DataLoader)-- the data loader for the training data
        test_loader (DataLoader) -- the data loader for the test/validation data
        loss_fn (Loss)           -- the loss function to use during training
        scheduler (Scheduler)    -- the learning rate scheduler (default: None)
        sch_step (bool)          -- whether to step the scheduler after each epoch (default: False)
    """

    def __init__(self, model, device, num_epochs, optimizer, train_loader, test_loader, loss_fn, scheduler=None, sch_step=False):
        self.model        = model
        self.device       = device
        self.num_epochs   = num_epochs
        self.optimizer    = optimizer
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.loss_fn      = loss_fn
        self.scheduler    = scheduler
        self.sch_step     = sch_step
        self.verbose      = True
    
    def train_epoch(self):
        self.model.train()
        for _, (inputs, labels) in enumerate(self.train_loader):
            
            # move data to device
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # reset loss for batch
            loss = 0

            def closure():
                nonlocal loss
                # zero gradient
                self.optimizer.zero_grad()

                # make prediction
                output = self.model(inputs)

                # compute loss
                loss = self.loss_fn(output, labels)

                # compute gradients
                loss.backward()
                return loss

            # adjust weights
            self.optimizer.step(closure)

    def test_nn(self, dataloader):
        self.model.eval()
        test_loss  = 0
        correct    = 0
        all_preds  = list()
        all_labels = list()

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs    = self.model(inputs)
                test_loss += self.loss_fn(outputs, labels).item()
                preds      = outputs.argmax(dim=1, keepdim=True)
                correct   += preds.eq(labels.view_as(preds)).sum().item()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss /= len(dataloader.dataset)
        accuracy   = correct / len(dataloader.dataset)

        all_preds   = np.array(all_preds)
        all_labels  = np.array(all_labels)
        f1_macro    = f1_score(all_labels, all_preds, average='macro')
        conf_matrix = confusion_matrix(all_labels, all_preds)

        return test_loss, f1_macro, accuracy, conf_matrix

    def train_nn(self):
        val_losses     = list()
        val_scores     = list()
        val_accuracies = list()
        tr_losses      = list()
        tr_scores      = list()
        tr_accuracies  = list()

        start = time.time()
        for epoch in range(self.num_epochs):
            
            # train model on this epoch
            self.train_epoch()

            # validate model
            test_loss, f1_macro, accuracy, conf_matrix = self.test_nn(self.test_loader)
            val_losses.append(test_loss)
            val_scores.append(f1_macro)
            val_accuracies.append(accuracy)

            train_loss, f1_train, acc_train, _ = self.test_nn(self.train_loader)
            tr_losses.append(train_loss)
            tr_scores.append(f1_train)
            tr_accuracies.append(acc_train)

            if self.scheduler is not None:
                if self.sch_step:
                    self.scheduler.step(test_loss)
                else:
                    self.scheduler.step()
            
            # show results of the current epoch
            if self.verbose:
                status_message = (
                    f"Epoch {epoch + 1}/{self.num_epochs} "
                    f"[Train] Loss: {train_loss:.7f} "
                    f"[Train] Accuracy: {acc_train:.4f} "
                    f"|| [Val] Loss: {test_loss:.7f} "
                    f"[Val] Accuracy: {accuracy:.4f}"
                )
                
                sys.stdout.write('\r' + status_message)
                sys.stdout.flush()

        end = time.time()

        print("\nTraining completed in {:.1f} seconds".format(end - start))

        return self.model, tr_losses, val_losses, range(1, self.num_epochs + 1), tr_scores, val_scores, tr_accuracies, val_accuracies, conf_matrix

    def save_model(self, file_name):
        if not file_name.endswith('.pth'): file_name += '.pth'
        torch.save(self.model.state_dict(), file_name)