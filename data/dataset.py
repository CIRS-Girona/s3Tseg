import os
import glob

from PIL import Image
from torch.utils.data import Dataset


class SonarDataset(Dataset):
    """Loads and stores (transformed) samples with their corresponding labels.
    Compatible with PyTorch DataLoader and Samplers.
    """

    def __init__(self, folder_path, transform=None):
        super().__init__()
        self.transform = transform
        self.img_files = glob.glob(os.path.join(folder_path,'images','*.tiff'))
        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(folder_path,'masks',os.path.basename(img_path)))

    def __getitem__(self, index):
        sample = (Image.open(self.img_files[index]).convert('L'), Image.open(self.mask_files[index]))
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.img_files)
