from pathlib import Path

from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from tqdm import tqdm


def calculate_class_weights(dataloader: DataLoader, class_label: str, num_classes: int):
    class_weights = [0] * num_classes

    sample_weights = [0 for _ in range(len(dataloader.dataset))]

    for data in tqdm(dataloader):
        for i, c in zip(data["idx"].tolist(), data[class_label].tolist()):
            class_weights[int(c)] += 1
            sample_weights[int(i)] = int(c)

    max_class_weight = max(class_weights)
    class_weights = [max_class_weight / class_weight for class_weight in class_weights]

    for i in range(len(sample_weights)):
        sample_weights[i] = class_weights[sample_weights[i]]

    return sample_weights, class_weights


class WeightedSamplerFromFile(WeightedRandomSampler):
    def __init__(self, filepath: str | Path, num_samples: int):
        sample_weights = self.read_sample_weights(filepath)

        super(WeightedSamplerFromFile, self).__init__(
            weights=sample_weights, num_samples=num_samples
        )

    def read_sample_weights(self, filepath: str | Path):
        with open(filepath, "r") as file:
            sample_weights = [float(line.strip()) for line in file]

        return sample_weights

    @staticmethod
    def write_sample_weights(filepath: str | Path, sample_weights):
        with open(filepath, "w") as file:
            for weight in sample_weights:
                file.write(f"{weight}\n")


class CustomSubsetSampler(Sampler):
    """
    Custom sampler that selects every nth item from the dataset.
    """

    def __init__(self, data_source, every_n_items: int):
        self.data_source = data_source
        self.indices = list(range(0, len(self.data_source), every_n_items))

    def __iter__(self):
        """
        Iterate over the selected indices to yield indices directly.
        """
        for idx in self.indices:
            yield idx

    def __len__(self):
        """
        Returns the number of indices selected.
        """
        return len(self.indices)
