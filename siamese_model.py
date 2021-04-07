import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import PIL.ImageOps    


class Siamese(nn.Module):
    ''' Siamese network to learn images representation'''
    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 10),  #64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  #64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),    #128@42*42
            nn.MaxPool2d(2),   #128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(), #128@18*18
            nn.MaxPool2d(2), #128@9*9
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),   #256@6*6
        )
        self.fully = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fully(x)
        return x

    def forward(self, x1, x2):
        output_1 = self.forward_one(x1)
        output_2 = self.forward_one(x2)
        return output_1, output_2

class SiameseDataset(Dataset):
    ''' Implements dataset creation for siamese network. Keeps track of selected images'''

    def __init__(self,df_paths_labels,length,chosen_labels=None,transform=None, p=0.3):
        self.df = df_paths_labels
        self.len_df = len(df_paths_labels)    
        if chosen_labels is not None:
            self.chosen_labels = chosen_labels
        else:
            self.chosen_labels = df_paths_labels.label.unique()
        self.length = length
        self.transform = transform
        self.fraction_same = p
        
    def __getitem__(self,index):
        path_1 = random.choice(self.df.loc[self.df.label.isin(self.chosen_labels)].path.values)
        label_1 = path_to_label(path_1)
        # self.df.iloc[random.randint(0,self.len_df),::]
        self.selected_images.append(path_1)

        # Dataset with a fraction p of positively labeled pairs
        same_label = random.random()
        same_label = int(same_label < self.fraction_same)

        if same_label:
            # Picks image from the same label as the first one
            path_2 = random.choice(self.df.loc[(self.df.label == label_1) & ~(self.df.path == path_1)].path.values)
            y = torch.from_numpy(np.array([0],dtype=np.float32))
        else:
            # Picks image from a different label
            path_2 = random.choice(self.df.loc[(self.df.label != label_1)].path.values)
            y = torch.from_numpy(np.array([1],dtype=np.float32))

        img_1 = Image.open(path_1).convert("RGB")
        img_2 = Image.open(path_2).convert("RGB")
        
        # Bools for data augmentation
        mirror_1, mirror_2, crop_1, crop_2 = random.randint(0,1), random.randint(0,1), random.randint(0,1), random.randint(0,1)

        # Avoids mirroring both images
        if int(mirror_1):
            img_1 = PIL.ImageOps.mirror(img_1)
        elif int(mirror_2):
            img_2 = PIL.ImageOps.mirror(img_2)

        # Avoids cropping both images
        proportion = 0.7
        if crop_1:
            x_size, y_size = img_1.size
            width, height = int(proportion*x_size), int(proportion*y_size)
            x_top_left, y_top_left = random.randint(0,x_size-width), random.randint(0,y_size-height)
            img_1 = img_1.crop((x_top_left, y_top_left, x_top_left + width, y_top_left + height))
        elif crop_2:
            x_size, y_size = img_2.size
            width, height = int(proportion*x_size), int(proportion*y_size)
            x_top_left, y_top_left = random.randint(0, x_size-width), random.randint(0, y_size-height)
            img_2 = img_2.crop((x_top_left, y_top_left, x_top_left + width, y_top_left + height))

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
        
        return img_1, img_2 , y
    
    def __len__(self):
        return self.length