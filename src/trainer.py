import torch
import json
from constants.control_variables import MODEL_NAME
from constants.paths import MODEL_ROOT
from validation_loss_checker import ValidationLossChecker
from utils import write_train_loss_to_file, save_model
from torch.nn import BCEWithLogitsLoss
from custom_loss_fns.focal_loss import FocalLoss
from custom_loss_fns.class_balanced_focal_loss import ClassBalancedFocalLoss



class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_loader,
        validation_loader,
        class_weights,
        NUM_EPOCHS,
    ):
        """Initialize the training loop

        Args:
            model (BaselineModel): the model to be rtained
            optimizer (Adam): Optimizer to use
            loss_fn (BCEWithLogitsLoss or FocalLoss): loss function to use
            train_loader (DataLoader): dataloader for training
            validation_loader (DataLoader): dataloader to compute validation loss for early stopping
            class_weights (Tensor): Tensor of class weights to be passed through when computing validation loss
            NUM_EPOCHS (int): max number of epochs to train
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.class_weights = class_weights
        self.NUM_EPOCHS = NUM_EPOCHS

    def train_model(self):
        """Train the model, computing validation and training loss at every epoch,
        saving the model each time the validation loss improves enough

        Returns:
            tuple: returns the  average train and validation losses for each epoch,
            as well as the epoch number to be saved to a file and to be plotted
        """

        validation_loss_checker = ValidationLossChecker(
            min_improvement=0.0001,
            epochs_to_wait=10,
            validation_loader=self.validation_loader,
            class_weights=self.class_weights,
            loss_fn=self.loss_fn,
        )

        print(type(self.loss_fn))

        if isinstance(self.loss_fn, BCEWithLogitsLoss):
            print(f"BCE Loss Function Pos Weights: {self.loss_fn.pos_weight}")
        elif isinstance(self.loss_fn, FocalLoss):
            print(f"Focal Loss Function Alphas : {self.loss_fn.alpha}")
        elif isinstance(self.loss_fn, ClassBalancedFocalLoss):
            print(f"Class Balanced Focal Loss betas: {self.loss_fn.beta}")

        train_losses = []  # list of loss values to plot
        epoch_list = []  # corresponding epoch of each loss value during training

        loss_so_far = 0
        print(self.model)

        for epoch in range(self.NUM_EPOCHS):

            current_batch_num = 0
            total_loss_for_epoch = 0
            self.model.train()
            for i, data in enumerate(self.train_loader):
                images, labels = data
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss_for_epoch += loss.item()
                current_batch_num += 1

                if i % 50 == 49:  # print the loss every 50 batches
                    loss_so_far = (
                        total_loss_for_epoch / current_batch_num
                    )  # avg loss up to current batch number
                    print(f"loss so far at batch {current_batch_num}: {loss_so_far}")

            # after each epoch during training, add the epoch number and loss value
            # to the list to be visualized
            epoch_average_loss = total_loss_for_epoch / (
                current_batch_num
            )  # average loss total so far up to the current epoch
            epoch_list.append(epoch + 1)
            train_losses.append(epoch_average_loss)

            print(
                f"Epoch {epoch+1} complete. Final Avg Loss for this epoch: {epoch_average_loss}"
            )  # average loss across all batches

            if validation_loss_checker.training_should_stop(self.model):
                print(f"NOT ENOUGH IMPROVEMENT FOUND, STOPPING TRAINING")
                validation_losses = validation_loss_checker.current_metrics
                write_train_loss_to_file(epoch_list, train_losses, validation_losses)

                # load model and calculate + save thresholds
                self.model = torch.load(MODEL_ROOT / f"{MODEL_NAME}.pt", weights_only=False)
                thresholds = validation_loss_checker.calculate_optimal_threshold(self.model)
                threshold_dict = {"thresholds": thresholds}
                with open(MODEL_ROOT / "thresholds.json", 'w') as threshold_file:
                    json.dump(threshold_dict, threshold_file)

                return (train_losses, validation_losses, epoch_list)

        save_model(self.model)
        validation_losses = validation_loss_checker.current_metrics
        write_train_loss_to_file(epoch_list, train_losses, validation_losses)

        # if training never stops still have to load the best model which may not be the latest one
        self.model = torch.load(MODEL_ROOT / f"{MODEL_NAME}.pt", weights_only=False)
        thresholds = validation_loss_checker.calculate_optimal_threshold(self.model)
        threshold_dict = {"thresholds": thresholds}
        with open(MODEL_ROOT / "thresholds.json", 'w') as threshold_file:
            json.dump(threshold_dict, threshold_file)

        print(f"training completed")

        return (train_losses, validation_losses, epoch_list)
