from torchvision import transforms
from torch.utils.data import random_split
from single_image_dataset import SingleImageDataset
import torch.nn as nn
import torch
from trainer import train_model, train_classifier
import utils
from random import shuffle
from torch.utils.data.dataset import Subset


class VGGTrain:
    '''
    Used to train vgg-like models
    '''
    def __init__(self, model, file_name_train, dir_name_train, file_name_val=None, dir_name_val=None, val_percent=5, regression=True, num_workers=4):
        '''
        The constructor for the vgg trainer. Calls build_extender which overrides the last layer of the model.
        :param model: The specified vgg-like model
        :param file_name_train: Train the model according to this file
        :param dir_name_train: Train the model according to this directory
        :param file_name_val: (Optional) Validate the model according to this file
        :param dir_name_val: (Optional) Validate the model according to this directory
        :param val_percent: (Optional) if not specified val_file, val_dir, percentage of training set to be used as val.
        :param regression: Use regression or classification
        :param num_workers: Number of workers for parallel computing
        '''
        self.main_model = model
        self.my_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((256,256)),
                transforms.CenterCrop((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        self.regression = regression
        self.dir_name = dir_name_train
        self.file_name = file_name_train
        total_dataset_1 = SingleImageDataset(file_name_train, dir_name_train, transform=self.my_transforms['train'], regression=regression)
        self.total_dataset_size = len(total_dataset_1)
        if file_name_val is None:
            total_dataset_2 = SingleImageDataset(file_name_train, dir_name_train, transform=self.my_transforms['val'], regression=regression)
            val_size = int(self.total_dataset_size * val_percent / 100)
            tmp_dssize = [self.total_dataset_size - val_size, val_size]
            tmp_idx = list(range(self.total_dataset_size))
            shuffle(tmp_idx)
            train_idx, val_idx = tmp_idx[:tmp_dssize[0]], tmp_idx[tmp_dssize[0]:]
            self.dataset = {'train': Subset(total_dataset_1, train_idx), 'val': Subset(total_dataset_2, val_idx)}
            self.dataset_size = {'train': tmp_dssize[0], 'val': tmp_dssize[1]}
        else:
            total_dataset_2 = SingleImageDataset(file_name_val, dir_name_val,
                                                 transform=self.my_transforms['val'], regression=regression)
            self.dataset = {'train': total_dataset_1, 'val': total_dataset_2}
            self.dataset_size = {'train': len(total_dataset_1), 'val': len(total_dataset_2)}
            self.dataloaders = {
                x: torch.utils.data.DataLoader(self.dataset[x], batch_size=32, shuffle=True, num_workers=num_workers)
                for x in ['train', 'val']}
        self.dataloaders = {x: torch.utils.data.DataLoader(self.dataset[x], batch_size=32, shuffle=True, num_workers=num_workers)
              for x in ['train', 'val']}
        self.build_extender(self.regression)

    def build_extender(self, regression):
        '''
        Overrides the last layer for feature extraction
        :param regression: Specifies if this is regression or not
        :return: Nothing
        '''
        for param in self.main_model.parameters():
            param.requires_grad = False
        num_ftrs = self.main_model.classifier[6].in_features
        fc1 = nn.Linear(num_ftrs, 1000)
        fc2 = nn.Linear(1000, 480)
        if regression:
            fc3 = nn.Linear(480, 2)
        else:
            fc3 = nn.Linear(480, 15)
        nn.init.xavier_normal_(fc1.weight)
        nn.init.xavier_normal_(fc2.weight)
        nn.init.xavier_normal_(fc3.weight)
        self.main_model.classifier[6] = nn.Sequential(
            fc1,
            nn.ReLU(),
            fc2,
            nn.ReLU(),
            fc3
        )

    # call this first
    def feat_extract(self, lr=0.001, epochs=5, top_k=1):
        '''
        Performs feature extraction
        :param lr: Learning rate
        :param epochs: Number of epochs
        :param top_k: If classification, top what?
        :return: The model with the best validation score, among with the actual score
        '''
        optimizer_ft = torch.optim.Adam(self.main_model.classifier[6].parameters(), lr=lr)
        if self.regression:
            criterion = utils.custom_loss
            return train_model(self.main_model, criterion=criterion, optimizer=optimizer_ft,
                             dataloaders=self.dataloaders, dataset_sizes=self.dataset_size,
                             num_epochs=epochs)
        else:
            criterion = nn.CrossEntropyLoss()
            return train_classifier(self.main_model, criterion=criterion, optimizer=optimizer_ft,
                                    dataloaders=self.dataloaders,
                                    dataset_sizes=self.dataset_size,
                                    num_epochs=epochs, top_k=top_k)

    def fine_tune(self, lr=0.0001, epochs=5, top_k=1):
        '''
        Fine tuning the model
        :param lr: Learning rate
        :param epochs: Number of epochs
        :param top_k: If classification, top what?
        :return: The model with the best validation score, among with the actual score
        '''
        for param in self.main_model.parameters():
            param.requires_grad = True
        optimizer_ft = torch.optim.Adam(self.main_model.parameters(), lr=lr)
        if self.regression:
            criterion = utils.custom_loss
            return train_model(self.main_model, criterion=criterion, optimizer=optimizer_ft,
                               dataloaders=self.dataloaders,
                               dataset_sizes=self.dataset_size,
                               num_epochs=epochs)
        else:
            criterion = nn.CrossEntropyLoss()
            return train_classifier(self.main_model, criterion=criterion, optimizer=optimizer_ft,
                                    dataloaders=self.dataloaders,
                                    dataset_sizes=self.dataset_size,
                                    num_epochs=epochs, top_k=top_k)
