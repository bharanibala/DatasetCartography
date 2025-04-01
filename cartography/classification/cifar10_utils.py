import os
import torch
from torchvision import datasets, transforms
from transformers.data.processors.utils import DataProcessor, InputExample

class CIFAR10Processor(DataProcessor):
    """Processor for the CIFAR-10 dataset."""

    # def __init__(self, data_dir, transform=None):
    #     self.data_dir = data_dir
    #     self.transform = transform if transform else transforms.ToTensor()

    def get_labels(self):
        """CIFAR-10 has 10 classes."""
        return [str(i) for i in range(10)]

    def _create_examples(self, dataset, set_type):
        """Creates examples from the dataset."""
        examples = []
        for idx, (image, label) in enumerate(dataset):
            guid = f"{set_type}-{idx}"
            if self.transform:
                image = self.transform(image)

            examples.append(InputExample(guid=guid, image_path=image, label=str(label)))
        return examples

    def get_examples(self, dataset, set_type):
        return self._create_examples(dataset, set_type=set_type)

    def get_train_examples(self, data_dir):
        """Load the train dataset from CIFAR-10."""
        train_set = datasets.CIFAR10(data_dir, train=True, download=True)
        return self.get_examples(train_set, set_type="train")

    def get_dev_examples(self, data_dir):
        """Load the validation (split from train) dataset from CIFAR-10."""
        # CIFAR-10 doesn't provide an explicit validation set, so we split from train
        train_set = datasets.CIFAR10(data_dir, train=True, download=True)
        train_size = int(0.8 * len(train_set))
        val_size = len(train_set) - train_size
        _, val_set = torch.utils.data.random_split(train_set, [train_size, val_size])
        return self.get_examples(val_set, set_type="dev")

    def get_test_examples(self, data_dir):
        """Load the test dataset from CIFAR-10."""
        test_set = datasets.CIFAR10(data_dir, train=False, download=True)
        return self.get_examples(test_set, set_type="test")