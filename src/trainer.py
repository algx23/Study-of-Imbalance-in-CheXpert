from validation_loss_checker import ValidationLossChecker
from utils import write_train_loss_to_file, save_model

class Trainer():
    def __init__(self, model, optimizer, loss_fn, train_loader, validation_loader, class_weights, NUM_EPOCHS):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.class_weights = class_weights
        self.NUM_EPOCHS = NUM_EPOCHS


    def train_model(self):

        validation_loss_checker = ValidationLossChecker(
            min_improvement=0.0001,
            epochs_to_wait=10,
            validation_loader = self.validation_loader,
            class_weights=self.class_weights
        )
        
        print(f"Loss Weights: {self.loss_fn.pos_weight}")

        train_losses = [] # list of loss values to plot
        epoch_list = [] # corresponding epoch of each loss value during training

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

                if i % 50 == 49: # print the loss every 50 batches
                    loss_so_far = total_loss_for_epoch / current_batch_num # avg loss up to current batch number
                    print(f"loss so far at batch {current_batch_num}: {loss_so_far}")

            # after each epoch during training, add the epoch number and loss value
            # to the list to be visualized
            epoch_average_loss = total_loss_for_epoch/(current_batch_num) # average loss total so far up to the current epoch
            epoch_list.append(epoch+1)
            train_losses.append(epoch_average_loss)

            print(f"Epoch {epoch+1} complete. Final Avg Loss for this epoch: {epoch_average_loss}") # average loss across all batches

            if validation_loss_checker.training_should_stop(self.model):
                print(f"NOT ENOUGH IMPROVEMENT FOUND, STOPPING TRAINING")
                validation_losses = validation_loss_checker.current_losses
                write_train_loss_to_file(epoch_list, train_losses, validation_losses)

                self.model.load_state_dict(torch.load(MODEL_ROOT / f"{MODEL_NAME}.pt"))

                return(train_losses, validation_losses, epoch_list)

        save_model(self.model)

        validation_losses = validation_loss_checker.current_losses
        write_train_loss_to_file(epoch_list, train_losses, validation_losses)


        print(f"training completed")

        return (train_losses, validation_losses, epoch_list)

