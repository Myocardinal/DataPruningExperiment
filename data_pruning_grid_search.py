import time
from typing import Tuple, List

import torch
import torchvision
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchmetrics import Accuracy
import pytorch_lightning as pl


class PruningDataset(Dataset):
    """
    Dataset wrapper that prunes a fraction of the data based on the loss values of the previous epoch.

    USAGE:
    - Before training starts: pruning_dataset = PruningDataset(train_dataset, p=0.2)
    - If we want next epoch to use the full dataset: pruning_dataset.set_next_epoch_as_full_train()
    - If we want to update the loss values seen: pruning_dataset.update_loss_values(indices, losses)
    - If we want to prune the next epoch using the loss values of the current epoch
        - Infobatch strategy randomly prunes 100 * p % of data that has loss < mean of losses
        - Full sort strategy prunes the bottom 100 * p % of data with the lowest losses
    """
    def __init__(self, dataset, p=0.2):
        """
        :param dataset: The original dataset to be pruned.
        :param p: A parameter that adjusts pruning rate.
        """
        # We assume that all instance variable tensors are stored on the CPU
        self.dataset: Dataset = dataset  # Original dataset
        self.p: float = p  # Pruning rate

        # Last seen losses of the data points in the dataset
        self.last_seen_losses: Tensor = torch.full((len(dataset),), float('inf'))  # 1-d Tensor of losses

        # Initialize the list of indices to be used in the next epoch, as all indices in original dataset
        remaining_indices_mask: Tensor = torch.ones(len(dataset), dtype=torch.bool)
        self._remaining_indices_cache = torch.where(remaining_indices_mask)[0]  # 1-d Tensor of indices

    def set_next_epoch_as_full_train(self):
        """
        Set the next epoch to use the full dataset without pruning.
        """
        remaining_indices_mask = torch.ones(len(self.last_seen_losses), dtype=torch.bool)
        self._remaining_indices_cache = torch.where(remaining_indices_mask)[0]

    def update_loss_values(self, indices, losses):
        """
        Update the loss values of the data points seen in the current epoch.
        :param indices: The indices of the data points evaluated in the current epoch.
        :param losses: The loss values of the data points evaluated in the current epoch.
        """
        self.last_seen_losses[indices] = losses.detach().cpu()

    def prune_next_epoch_using_infobatch(self):
        """
        Prune the next epoch using the infobatch strategy.
        """
        # Compute the mean of the last seen losses among all data points
        losses_mean: float = torch.mean(self.last_seen_losses).item()

        # Data remains in last epoch if loss is greater than mean or randomly selected with probability (1-p)
        randomly_remain_mask: Tensor = torch.gt(torch.rand(len(self.last_seen_losses)), self.p)
        large_losses_mask: Tensor = torch.greater_equal(self.last_seen_losses, losses_mean)
        remaining_indices_mask: Tensor = torch.logical_or(large_losses_mask, randomly_remain_mask)

        # Update the list of indices that remain in the pruned dataset
        self._remaining_indices_cache = torch.where(remaining_indices_mask)[0]

    def prune_next_epoch_using_full_sort(self):
        """
        Prune the next epoch using the full sort strategy.
        """
        # Compute the threshold between large and small losses
        random_subset_size: int = 5000
        random_subset_indices: Tensor = torch.randperm(len(self.last_seen_losses))[:random_subset_size]
        random_subset_losses: Tensor = self.last_seen_losses[random_subset_indices]
        threshold: float = torch.quantile(random_subset_losses, self.p).item()

        # Data remains in last epoch if loss is greater than threshold
        remaining_indices_mask: Tensor = torch.greater_equal(self.last_seen_losses, threshold)

        # Update the list of indices that remain in the pruned dataset
        self._remaining_indices_cache = torch.where(remaining_indices_mask)[0]

    def __len__(self):
        """
        :return: The number of data points in the pruned dataset.
        """
        return len(self._remaining_indices_cache)

    def __getitem__(self, idx):
        """
        :param idx: The index of the data point in the pruned dataset.
        :return: The data point's X and y values, and the index of the data point in the original dataset.
        """
        actual_idx = self._remaining_indices_cache[idx].item()
        x, y = self.dataset[actual_idx]
        return x, y, actual_idx


class BasicBlock(nn.Module):
    """
    A basic block of the ResNet architecture.
    It consists of two convolutional layers with batch normalization and ReLU activation.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int=1, width_scaling_factor: float=1.0):
        """
        Initialize the basic block.
        :param in_channels: The number of input channels to the block.
        :param out_channels: The number of output channels of the block.
        :param stride: The stride of the first convolutional layer.
        :param width_scaling_factor: A scaling factor for the number of channels in the block.
        """
        super(BasicBlock, self).__init__()
        # Compute the number of channels after scaling, but keep it at least 8 channels
        scaled_out_channels = max(8, int(out_channels * width_scaling_factor))

        # Construct a block with two convolutional layers, two batch norms, and ReLU activation
        self.conv1 = nn.Conv2d(in_channels, scaled_out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(scaled_out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(scaled_out_channels, scaled_out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(scaled_out_channels)

        # Construct a shortcut connection if the number of channels changes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != scaled_out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, scaled_out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(scaled_out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass (inference time) logic for a ResNet block.
        :param x: The input tensor to the block.
        :return: The output tensor of the block.
        """
        # Forward pass consists of conv -> bn -> relu -> conv -> bn -> shortcut -> relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ScalableResNet(pl.LightningModule):
    """
    A scalable ResNet model with adjustable depth and width scaling, that works with PruningDataset.
    """
    def __init__(self, depth_scaling_factor, width_scaling_factor):
        super(ScalableResNet, self).__init__()
        self.in_channels = 16  # Keep base width constant at 16 channels
        self.outputs = []

        # Initial convolution
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # Define the layers (equivalent to ResNet16 when scaling factors are 1)
        num_blocks_per_layer = max(1, int(3 * depth_scaling_factor))  # Base ResNet16 has 3 layers, with 3 blocks each
        self.layer1 = self._make_layer(16, num_blocks_per_layer, stride=1, width_scaling_factor=width_scaling_factor)
        self.layer2 = self._make_layer(32, num_blocks_per_layer, stride=2, width_scaling_factor=width_scaling_factor)
        self.layer3 = self._make_layer(64, num_blocks_per_layer, stride=2, width_scaling_factor=width_scaling_factor)

        # Final classifier
        final_out_channels = max(8, int(64 * width_scaling_factor))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(final_out_channels, 100)

        # Criterion and metrics
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.accuracy = Accuracy(task='multiclass', num_classes=100, average='macro', top_k=1)

    def _make_layer(self, out_channels, num_blocks, stride, width_scaling_factor):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, width_scaling_factor))
        self.in_channels = max(8, int(out_channels * width_scaling_factor))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, stride=1, width_scaling_factor=width_scaling_factor))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Ensure x is moved to the same device as the model
        x = x.to(self.device)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        """
        :return: The optimizer to use for training, which is Adam with default lr.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch: List[Tuple[Tensor, Tensor, int]], batch_idx: int):
        """
        Perform training on a minibatch. Note that we assume training uses a DataLoader created using a PruningDataset.
        Hence, the DataLoader should return a collection of (x, y, indices) tuples, and not just (x, y) tuples.
        We assume that the batch lives in CPU.
        :param batch: The batch of data to train on.
        :param batch_idx: The index of the batch in the dataset.
        :return: The loss value of the batch.
        """
        # Unpack the batch
        x, y, indices = batch

        # Send (x, y) to device to run the forward pass
        x: Tensor = x.to(self.device)
        y: Tensor = y.to(self.device)
        y_hat: Tensor = self(x)

        # Compute mean loss, which will be used for the gradient step
        loss_vector: Tensor = self.criterion(y_hat, y)
        loss = loss_vector.mean()

        # Return the losses and indices used for pruning logic
        return {'loss': loss, 'indices': indices, 'loss_vector': loss_vector}

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation (inference) step on a minibatch. Similar to training_step, we assume DataLoader
        uses a PruningDataset.
        :param batch: The batch of data to validate on.
        :param batch_idx: The index of the batch in the dataset.
        :return: The loss value of the batch.
        """
        # Unpack the batch
        x: Tensor
        y: Tensor
        x, y, _ = batch

        # Send (x, y) to device to run the forward pass
        x: Tensor = x.to(self.device)
        y: Tensor = y.to(self.device)
        y_hat: Tensor = self(x)

        # Compute the loss value of the batch
        loss = self.criterion(y_hat, y).mean()
        return {'loss': loss}


class ScalablePrunableResNet(ScalableResNet):
    """
    A scalable ResNet model that supports pruning of the training dataset.
    """
    def __init__(self, depth_scaling_factor: float, width_scaling_factor: float, pruning_method: str):
        super(ScalablePrunableResNet, self).__init__(depth_scaling_factor, width_scaling_factor)
        self.pruning_method = pruning_method
        self.outputs = []

    ### METHODS DEFINED IN THE RESNET CLASS

    def forward(self, x):
        return ScalableResNet.forward(self, x)

    def configure_optimizers(self):
        return ScalableResNet.configure_optimizers(self)

    ### METHODS FOR TRAINING & PRUNING LOGIC

    def training_step(self, batch: List[Tuple[Tensor, Tensor, int]], batch_idx: int):
        ret = ScalableResNet.training_step(self, batch, batch_idx)
        self.outputs.append({'loss_vector': ret['loss_vector'], 'indices': ret['indices']})
        return {'loss': ret['loss']}

    def validation_step(self, batch, batch_idx):
        return ScalableResNet.validation_step(self)

    def on_train_epoch_end(self):
        """
        At the end of each training epoch, update last seen losses and prune the dataset.
        """
        match self.pruning_method:
            case 'infobatch':
                # Concatenate the losses and indices from all batches in the epoch
                indices: List[Tensor] = [output['indices'] for output in self.outputs]
                losses: List[Tensor] = [output['loss_vector'].detach().cpu() for output in self.outputs]
                indices_all: Tensor = torch.cat(indices)
                losses_all: Tensor = torch.cat(losses)

                # Update the loss values of the data points in the dataset
                dataset: PruningDataset = self.trainer.train_dataloader.dataset
                dataset.update_loss_values(indices_all, losses_all)

                # Prune the next epoch using the infobatch strategy
                dataset.prune_next_epoch_using_infobatch()

                # Clear the outputs list for the next epoch
                for output in self.outputs:
                    del output['loss_vector'], output['indices']
                self.outputs = []

            case 'full_sort':
                full_train_epoch_interval = 5
                # Every several epochs, train on the full dataset without pruning
                if self.current_epoch % full_train_epoch_interval == full_train_epoch_interval-1:
                    # Set the next epoch to use the full dataset without pruning
                    dataset: PruningDataset = self.trainer.train_dataloader.dataset
                    dataset.set_next_epoch_as_full_train()
                # Then, prune the dataset using the full sort strategy
                elif self.current_epoch % full_train_epoch_interval == 0:
                    # Concatenate the losses and indices from all batches in the epoch
                    indices: List[Tensor] = [output['indices'] for output in self.outputs]
                    losses: List[Tensor] = [output['loss_vector'].detach().cpu() for output in self.outputs]
                    indices_all: Tensor = torch.cat(indices)
                    losses_all: Tensor = torch.cat(losses)

                    # Update the loss values of the data points in the dataset
                    dataset: PruningDataset = self.trainer.train_dataloader.dataset
                    dataset.update_loss_values(indices_all, losses_all)

                    # Prune the next epoch using the full sort strategy
                    dataset.prune_next_epoch_using_full_sort()

                    # Clear the outputs list for the next epoch
                    for output in self.outputs:
                        del output['loss_vector'], output['indices']
                    self.outputs = []
                else:
                    return


            case _:
                raise ValueError(f'Pruning method {self.pruning_method} not yet supported for PrunableModel.')


def train_scalable_resnet(
        train_dataset: Dataset,
        test_dataset: Dataset,
        depth_scaling_factor: float,
        width_scaling_factor: float,
        pruning_method: str,
        pruning_rate: float,
        batch_size: int,
        max_epochs: int,
        gpu_id: int
    ) -> Tuple[str, str, str, str]:
    """
    Trains a prunable, scalable ResNet model on a given dataset.
    :param train_dataset: The training dataset.
    :param test_dataset: The test dataset.
    :param depth_scaling_factor: The scaling factor for the depth of the ResNet model.
    :param width_scaling_factor: The scaling factor for the width of the ResNet model.
    :param pruning_method: The method to use for pruning the dataset.
    :param pruning_rate: The fraction of the dataset to prune at the beginning of each epoch.
    :param batch_size: The batch size to use for training.
    :param max_epochs: The maximum number of epochs to train the model.
    :param gpu_id: The ID of the GPU to use for training.
    :return: The total train latency (s), peak memory use (byte), final train accuracy, and final test accuracy.
    """
    # Set device to the specified GPU and clear memory for consistency
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()

    # Wrap the original dataset with PruningDataset for training purposes
    pruning_dataset = PruningDataset(train_dataset, p=pruning_rate)
    train_loader = DataLoader(pruning_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Create DataLoader for testing purposes
    train_base_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize and compile the model
    if pruning_rate > 0:
        model = ScalablePrunableResNet(depth_scaling_factor, width_scaling_factor, pruning_method)
    else:
        model = ScalableResNet(depth_scaling_factor, width_scaling_factor)
    torch.compile(model)

    try:
        # Performance metrics
        start_time = time.time()
        max_memory = 0

        # Initialize the trainer for full training
        trainer = pl.Trainer(max_epochs=max_epochs, log_every_n_steps=10, accelerator='gpu', devices=[gpu_id])

        # Train the model
        trainer.fit(model, train_loader)

        # Log GPU memory usage and total latency
        max_memory: int = max(max_memory, torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        end_time = time.time()
        total_time: float = end_time - start_time

        # Evaluate final train set accuracy
        model.eval()  # Set model to evaluation mode
        train_accuracy: float = compute_test_accuracy(model, train_base_loader)
        test_accuracy: float = compute_test_accuracy(model, test_loader)

    # Catch out-of-memory errors during training
    except RuntimeError as e:
        if 'out of memory' in str(e):
            total_time: str = '-'
            max_memory: str = 'OOM'
            train_accuracy: str  = '-'
            test_accuracy: str = '-'
        else:
            raise e

    return str(total_time), str(max_memory), str(train_accuracy), str(test_accuracy)


def compute_test_accuracy(model, test_loader):
    """
    Compute the test accuracy of a model on a given test dataset.
    :param model: The model to evaluate.
    :param test_loader: The DataLoader for the test dataset. It should return (x, y) tuples.
    :return: The test accuracy of the model on the test dataset.
    """
    with torch.no_grad():
        model.accuracy.reset()
        for batch in test_loader:
            # Unpack the batch
            x: Tensor
            y: Tensor
            x, y = batch
            # Get model predictions and update accuracy
            predictions = model(x)
            model.accuracy.update(predictions.argmax(dim=1), y)
        return model.accuracy.compute().item()



def grid_search_experiment():
    """
    Performs a series of experiments on combinations of scaling factors, pruning rates, and batch sizes.
    Then, logs the results as CSV-compatible format to a text file.
    """

    # Grid Search Parameters
    scaling_factors = [1.0, 1.1, 1.2, 1.3, 1.4]  # Depth scaling factors
    #width_scaling_factors = [1.0, 1.1, 1.2, 1.3, 1.4]  # Width scaling factors
    pruning_rates = [0.0, 0.1, 0.2, 0.3]  # Pruning rates
    batch_sizes = [2000]  # Batch sizes
    pruning_methods = ['full_sort']

    # Parameters set constant for all configs
    gpu_id_to_use = 0  # GPU ID to use for training
    max_epochs = 30  # Maximum number of epochs to train the model

    # Log filename
    log_filename = 'grid_search.txt'

    # Write header to log file
    with open(log_filename, mode='a') as file:
        file.write('depth_scaling_factor,width_scaling_factor,pruning_rate,batch_size,total_time,max_memory,train_accuracy,test_accuracy\n')

    # Load CIFAR100 dataset's train & test split, with standard transform
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Run experiments doing a grid search
    for width_scaling_factor in scaling_factors:
        for depth_scaling_factor in scaling_factors:
            for pruning_rate in pruning_rates:
                for batch_size in batch_sizes:
                    for pruning_method in pruning_methods:
                        # Train the model and get stats
                        total_time, max_memory, train_accuracy, test_accuracy = train_scalable_resnet(
                            train_dataset, test_dataset, depth_scaling_factor, width_scaling_factor, pruning_method,
                            pruning_rate, batch_size, max_epochs=max_epochs, gpu_id=gpu_id_to_use
                        )
                        # Append results to log file
                        with open(log_filename, mode='a') as file:
                            file.write(
                                f'{depth_scaling_factor},{width_scaling_factor},{pruning_rate},{batch_size},{total_time},{max_memory},{train_accuracy},{test_accuracy}\n')

if __name__ == "__main__":
    grid_search_experiment()
