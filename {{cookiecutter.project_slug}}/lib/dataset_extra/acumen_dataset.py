import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class AcumenDataset(Dataset):
	def __init__(self, args, kind = 'train', data_transforms = None):
		self.args = args
		self.kind = kind
		self.transforms = transforms

 @classmethod
    def train_dataloader(cls, args, annotations = None):
        dataset = cls(args = args, kind = 'train')

        return DataLoader(
            dataset,
            num_workers = 15,
            pin_memory = True,
            batch_size = args.batch_size
        )

    @classmethod
    def val_dataloader(cls, args, annotations = None):
        dataset = cls(args = args, data_transforms = None, annotations = annotations, kind = 'val')

        return DataLoader(
            dataset,
            batch_size = args.batch_size,
            shuffle = False,
            num_workers = 15,
            pin_memory = True,
        )

