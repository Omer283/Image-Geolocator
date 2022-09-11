from torch.utils.data import Dataset
from torch import is_tensor
from torch import tensor
import pandas as pd
import os
from PIL import Image
from torchvision import transforms


class SingleImageDataset(Dataset):
    '''
    Dataset for retrieving geolocated images
    '''
    def __init__(self, csv_file, dir, transform=None, regression=True):
        '''
        Constructor
        :param csv_file: Csv file of the dataset's images
        :param dir: Directory of the dataset's images
        :param transform: Transformations applied to the input
        :param regression: Should the dataset be return [lat, long] or class
        '''
        self.dir = dir
        self.transform = transform
        self.image_data = pd.read_csv(csv_file)
        self.reg = regression

    def __len__(self):
        '''
        :return: Amount of images in dataset
        '''
        return len(self.image_data)

    def __getitem__(self, idx):
        '''
        Returns the idx's image, along with its tag
        :param idx: index of queried image
        :return: The image, along with [lat, long] or class
        '''
        if is_tensor(idx):
            idx = idx.tolist()
        img_path = os.path.join(self.dir, self.image_data.iloc[idx, 0])
        image = Image.open(img_path)
        lat, long = self.image_data.iloc[idx, 1], self.image_data.iloc[idx, 2]
        num = self.image_data.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        if self.reg:
            return image, tensor([lat, long])
        else:
            return image, tensor(num)