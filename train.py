import time
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torch import optim
from random import choice, shuffle, sample
from torch.autograd import Variable

from loss import ContrastiveLoss, contrastive_batch_loss
from siamese_model import Siamese, SiameseDataset

EPOCHS = 10
net = Siamese()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.00005)
margin = 2.0
negative_mining = int(batch_size*0)

start = time.time()
for epoch in range(EPOCHS):
    for i, batch in enumerate(siamese_dataloader):
        img_0, img_1 , label = batch
        optimizer.zero_grad()

        with torch.no_grad():
            # The network is only trained on the top negative_mining examples
            temp_output_1, temp_output_2 = net(img_0,img_1)
            temp_loss_contrastive = contrastive_batch_loss(temp_output_1,temp_output_2,label)
            loss, indexes = torch.sort(temp_loss_contrastive, dim=0)

        indexes = indexes[-negative_mining:]
        output_1,output_2 = net(img_0[indexes].squeeze(),img_1[indexes].squeeze())
        loss_contrastive = criterion(output_1,output_2,label[indexes])
        loss_contrastive.backward()
        optimizer.step()

        print(f"\r Epoch number {epoch+1}/{EPOCHS}, batch number {i+1}/{int(data.length/batch_size)}, current loss {loss_contrastive.item(): .5}", end='')
        loss_history.append(loss_contrastive.item())        
end = time.time()
print(f'\n Training time {end-start: .5}s, that is {int((end-start)/3600)}h{int(60*(end-start)//3600)}min')
plt.plot(loss_history)
plt.show()