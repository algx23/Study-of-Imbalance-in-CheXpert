import torch
import json
from RunPathHolder import RunPathHolder
from validation_metric_checker import ValidationMetricCalculator
from utils import write_train_loss_to_file, save_model
from torch.nn import BCEWithLogitsLoss
from custom_loss_fns.focal_loss import FocalLoss
from custom_loss_fns.class_balanced_focal_loss import ClassBalancedFocalLoss

from torchvision.transforms.v2 import MixUp


class Trainer:
    def __init__(
        self,
        path_holder: RunPathHolder,
        model,
        optimizer,
        loss_fn: BCEWithLogitsLoss | FocalLoss | ClassBalancedFocalLoss,
        train_loader,
        validation_loader,
        NUM_EPOCHS: int,
        use_mixup=False,
        threshold=[0.3] * 13,
    ):
        """Initialize the training loop

        Args:
            model (BaselineModel): the model to be rtained
            optimizer (Adam): Optimizer to use
            loss_fn (BCEWithLogitsLoss | FocalLoss): loss function to use
            train_loader (DataLoader): dataloader for training
            validation_loader (DataLoader): dataloader to compute validation loss for early stopping
            class_weights (Tensor): Tensor of class weights to be passed through when computing validation loss
            NUM_EPOCHS (int): max number of epochs to train
        """
        self.path_holder = path_holder
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.NUM_EPOCHS = NUM_EPOCHS
        self.use_mixup = use_mixup
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_path = (
            self.path_holder.model_root / f"{self.path_holder.model_name}.pt"
        )
        self.loss_filepath = (
            self.path_holder.train_data_path / "avg_train_loss_val_prauc_per_epoch.csv"
        )

    def train_model(self):
        """Train the model, computing validation and training loss at every epoch,
        saving the model each time the validation loss improves enough

        Returns:
            tuple: returns the  average train and validation losses for each epoch,
            as well as the epoch number to be saved to a file and to be plotted
        """

        validation_metric_checker = ValidationMetricCalculator(
            best_model_save_path=self.model_path,
            min_improvement=0.0001,
            epochs_to_wait=10,
            validation_loader=self.validation_loader,
            loss_fn=self.loss_fn,
        )

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
                images = images.to(self.device)
                labels = labels.to(self.device)

                if self.use_mixup:
                    mixup = MixUp(num_classes=13)
                    images, labels = mixup(images, labels)

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

            if validation_metric_checker.training_should_stop(self.model):
                print(f"NOT ENOUGH IMPROVEMENT FOUND, STOPPING TRAINING")
                validation_ap = validation_metric_checker.current_metrics
                write_train_loss_to_file(
                    epoch_list,
                    train_losses,
                    validation_ap,
                    loss_file_path=self.loss_filepath,
                )

                # load model and calculate + save thresholds
                self.model = torch.load(
                    self.model_path,
                    weights_only=False,
                    map_location=self.device,
                )

                threshold_filepath = self.path_holder.model_root / "thresholds.json"
                if self.threshold == "optimal":
                    thresholds = validation_metric_checker.calculate_optimal_threshold(
                        self.model
                    )
                    threshold_dict = {"thresholds": thresholds}
                    with open(threshold_filepath, "w") as threshold_file:
                        json.dump(threshold_dict, threshold_file)

                return

        save_model(self.model, self.model_path)
        validation_ap = validation_metric_checker.current_metrics

        write_train_loss_to_file(
            epoch_list, train_losses, validation_ap, loss_file_path=self.loss_filepath
        )

        # if training never stops still have to load the best model which may not be the latest one
        self.model = torch.load(
            self.model_path,
            weights_only=False,
            map_location=self.device,
        )
        if self.threshold == "optimal":
            thresholds = validation_metric_checker.calculate_optimal_threshold(
                self.model
            )
            threshold_filepath = self.path_holder.model_root / "thresholds.json"
            threshold_dict = {"thresholds": thresholds}
            with open(threshold_filepath, "w") as threshold_file:
                json.dump(threshold_dict, threshold_file)

        print(f"training completed")

        return
