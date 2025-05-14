import re
from typing import Any

import torch
from torch.utils.data.dataloader import default_collate


class CollateBase:
    def __init__(
        self,
        item_keys: str | list[str] = ".*",
        delete_original=False,
        exclude_keys: list[str] = ["event_id"],
    ):
        self.item_keys = item_keys if isinstance(item_keys, list) else [item_keys]
        self.item_keys = [re.compile(key) for key in self.item_keys]
        self.delete_original = delete_original
        self.item_keys_cached = []
        self.exclude_keys = exclude_keys if exclude_keys else []

        # # Check if MPS is available
        # self.use_mps = torch.backends.mps.is_available()

        # if self.use_mps:
        #     logger.info("MPS is available. Using float32 for tensors.")

    def __call__(
        self, items_list: list[dict[str, Any]] | dict[str, Any]
    ) -> dict[str, Any]:

        assert len(items_list) > 0, "items_list must have at least one item"

        if isinstance(items_list, list):
            items = {
                key: [item[key] for item in items_list if not self.is_excluded(key)]
                for key in items_list[0].keys()
            }
        else:
            items = {
                key: value
                for key, value in items_list.items()
                if not self.is_excluded(key)
            }

        keys = self.match_keys(list(items.keys()))

        items_new = self.do_collate({key: items[key] for key in keys})
        if self.delete_original:
            for key in keys:
                del items[key]

        for key in items_new.keys():
            items[key] = items_new[key]

        return items

    def match_keys(self, keys: list[str]) -> list[str]:
        if not self.item_keys_cached:
            for key in keys:
                for item_key in self.item_keys:
                    if re.match(item_key, key):
                        self.item_keys_cached.append(key)
                        break

        return self.item_keys_cached

    def do_collate(self, item: dict[str, Any]):
        raise NotImplementedError

    def is_excluded(self, key: str) -> bool:
        """
        Checks if the key matches any of the exclude patterns.
        """
        for exclude_key in self.exclude_keys:
            if re.search(exclude_key, key):
                return True
        return False

    def ensure_float32(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is of dtype float32 if MPS is being used."""
        if self.use_mps and tensor.dtype == torch.float64:
            return tensor.to(dtype=torch.float32)
        return tensor


class BatchCollate(CollateBase):
    def do_collate(self, items):
        """
        Collate the items into a batch.

        items: dict[str, Any]
        """
        result = {
            key: (
                default_collate(items[key]).float()
                if not isinstance(items[key], torch.Tensor)
                else items[key]
            )
            for key in items.keys()
        }
        return result


class DeleteKeys(CollateBase):
    def __init__(self, keys: list[str]):
        super().__init__(keys, True)

    def do_collate(self, items):
        return {}


class ListCollate(CollateBase):
    def __init__(
        self, collates: list[CollateBase], item_keys=".*", delete_original=False
    ):
        super().__init__(item_keys, delete_original)
        self.collates = collates

    def do_collate(self, items):
        for collate in self.collates:
            items = collate(items)

        return items


class MaxCollate(CollateBase):
    def do_collate(self, items):
        return {key: torch.max(item, dim=1)[0] for key, item in items.items()}


class ConcatenateCollate(CollateBase):
    def __init__(self, new_key: str, dim=2, item_keys=".*", delete_original=True):
        super().__init__(item_keys, delete_original)

        self.new_key = new_key
        self.dim = dim

    def do_collate(self, items):
        item_list = [items[key] for key in items.keys()]
        return {self.new_key: torch.stack(item_list, dim=self.dim)}


class CombineCollate(CollateBase):
    def __init__(self, new_key: str, dim=1, item_keys=".*", delete_original=True):
        super().__init__(item_keys, delete_original)

        self.new_key = new_key
        self.dim = dim

    def do_collate(self, items):

        # Collect the tensors corresponding to the selected keys
        item_list = [items[key] for key in items.keys()]

        # Stack the tensors along the specified dimension
        stacked_items = torch.stack(item_list, dim=self.dim)

        # Take the maximum along the stacking dimension
        combined = torch.max(stacked_items, dim=self.dim)[0]

        return {self.new_key: combined}
