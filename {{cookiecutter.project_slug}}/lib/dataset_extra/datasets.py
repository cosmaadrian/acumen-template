from torch.utils.data import Dataset
from torchvision import datasets
import torch
from torchvision.transforms import Compose, ToTensor

class MnistDataset(Dataset):
    @staticmethod
    def train_dataloader(args):
        train_data = datasets.MNIST(
            root = 'data',
            train = True,                         
            transform = ToTensor(),
            download = True,            
        )

        return torch.utils.data.DataLoader(MnistDataset(args, train_data), batch_size = args.batch_size, shuffle = True, num_workers = 1)
    
    def __init__(self, args, train_data):
        self.args = args
        self.train_data = train_data

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        image, label = self.train_data[idx]

        return {
            'image': image,
            'labels': label
        }
