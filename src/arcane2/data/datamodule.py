from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class ParsedDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        num_workers,
        shuffle=True,
        train_sampler=None,
        val_sampler=None,
        test_sampler=None,
        train_collate_fn=None,
        val_collate_fn=None,
        test_collate_fn=None,
        persisten_workers=True,
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler

        self.train_collate_fn = train_collate_fn
        self.val_collate_fn = val_collate_fn
        self.test_collate_fn = test_collate_fn

        self.persisten_workers = persisten_workers

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle if self.train_sampler is None else False,
            sampler=self.train_sampler,
            collate_fn=self.train_collate_fn,
            persistent_workers=self.persisten_workers,
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.val_sampler,
            collate_fn=self.val_collate_fn,
            persistent_workers=self.persisten_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.test_sampler,
            collate_fn=self.test_collate_fn,
            persistent_workers=self.persisten_workers,
        )

    def __repr__(self):
        return f"ParsedDataModule:\nTrain: {self.train_dataset}\nVal: {self.val_dataset}\nTest: {self.test_dataset}"
