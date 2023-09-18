import torch
from torch.utils.data import Dataset, DataLoader

class AcumenDataset(Dataset):
    def __init__(self, args, kind = 'train', data_transforms = None, annotations = None):
        self.args = args
        self.kind = kind
        self.annotations = annotations
        self.data_transforms = data_transforms

    @classmethod
    def train_dataloader(cls, args, annotations = None):
        dataset = cls(args = args, kind = 'train', annotations = annotations)

        return DataLoader(
            dataset,
            num_workers = args.environment.extra_args.num_workers if not args.debug else 1,
            pin_memory = True,
            shuffle = True,
            batch_size = args.batch_size
        )

    @classmethod
    def val_dataloader(cls, args, annotations = None, kind = 'val'):
        dataset = cls(args = args, data_transforms = None, annotations = annotations, kind = kind)

        return DataLoader(
            dataset,
            batch_size = args.eval_batch_size,
            shuffle = False,
            num_workers = args.environment.extra_args.num_workers if not args.debug else 1,
            pin_memory = True,
        )

