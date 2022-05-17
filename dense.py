import torch.nn as nn
import numpy as np
import torch.optim as optim
import utils
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.models import densenet121

import time
import os

name = 'densenet_all'

all_data = True
n_examples = 20000

n_split = 0.1

n_epochs = 20

# number of cpu cores, usually
num_workers = 8

batch_size = 64

# optimizer
learning_rate = 1e-3
weight_decay = 1e-4

# learning rate scheduler
step_size = 5
gamma = 0.7

transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize([123.1379],[77.6652]),
    transforms.Lambda(
        lambda x: x.repeat(3,1,1)
    )
])

cuda = torch.cuda.is_available()

print(cuda)
print(torch.cuda.device_count())

class DenseNet(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(DenseNet,self).__init__()
        
        # self.c1 = nn.Conv2d(5, 3, kernel_size=3)
        self.densenet121 = densenet121(pretrained=False)       
        num_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_features, 14),
            nn.Tanh()
        )

    def forward(self, x):
        # x = F.leaky_relu(F.max_pool2d(self.c1(x),2))
        x = self.densenet121(x)
        return x

if all_data:
    sample = 'all'
else:
    sample = 'random'

frontal_dataset = utils.CheXDataset(csv_file='train.csv',
                                    data_dir='/groups/CS156b/data/',
                                    n_samples=n_examples,
                                    sample_type=sample,
                                    xray_type='frontal',
                                    pathology='all',
                                    fill_type='zero',
                                    transform=transform
                                    )

lateral_dataset = utils.CheXDataset(csv_file='train.csv',
                                    data_dir='/groups/CS156b/data/',
                                    n_samples=n_examples,
                                    sample_type=sample,
                                    xray_type='lateral',
                                    pathology='all',
                                    fill_type='zero',
                                    transform=transform
                                    )

print(len(frontal_dataset))
print(len(lateral_dataset))

n_train = int((1-n_split)*len(frontal_dataset))
n_val = len(frontal_dataset) - n_train

frontal_train, frontal_val = random_split(frontal_dataset, [n_train, n_val])

n_train = int((1-n_split)*len(lateral_dataset))
n_val = len(lateral_dataset) - n_train

lateral_train, lateral_val = random_split(lateral_dataset, [n_train, n_val])

print(len(frontal_train), len(frontal_val))
print(len(lateral_train), len(lateral_val))

frontal_train_dl = DataLoader(frontal_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True
                              )

lateral_train_dl = DataLoader(lateral_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True
                              )

frontal_val_dl = DataLoader(frontal_val,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True
                            )

lateral_val_dl = DataLoader(lateral_val,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True
                            )

frontal_model = DenseNet()
lateral_model = DenseNet()

frontal_model = nn.DataParallel(frontal_model)
lateral_model = nn.DataParallel(lateral_model)

if cuda:
    frontal_model.cuda()
    lateral_model.cuda()

loss_fn = nn.MSELoss(reduction='sum')

frontal_optimizer = optim.Adam(frontal_model.parameters(),
                               lr=learning_rate,
                               weight_decay=weight_decay
                               )

lateral_optimizer = optim.Adam(lateral_model.parameters(),
                               lr=learning_rate,
                               weight_decay=weight_decay
                               )

frontal_scheduler = optim.lr_scheduler.StepLR(frontal_optimizer,
                                              step_size=step_size,
                                              gamma=gamma
                                              )

lateral_scheduler = optim.lr_scheduler.StepLR(lateral_optimizer,
                                              step_size=step_size,
                                              gamma=gamma
                                              )

loss_fn = nn.MSELoss(reduction='sum')

frontal_losses = np.zeros(n_epochs)
lateral_losses = np.zeros(n_epochs)

frontal_validation = np.zeros(n_epochs)
lateral_validation = np.zeros(n_epochs)

for epoch in range(n_epochs):
    print('='*25)
    print('Epoch {}/{}:'.format(epoch+1, n_epochs))
    print('='*25)
    
    start_time = time.time()
        
    print('-'*25)
        
    frontal_model.train()
        
    for i, data in enumerate(frontal_train_dl):
        images, labels = data
        if cuda:
            images, labels = images.cuda(), labels.cuda()
                
        frontal_optimizer.zero_grad()
        output = frontal_model(images)
        loss = loss_fn(output, labels)
                
        loss.backward()
        frontal_optimizer.step()
            
        frontal_losses[epoch] += loss.item()
        
    lateral_model.train()
        
    for i, data in enumerate(lateral_train_dl):
        images, labels = data
        if cuda:
            images, labels = images.cuda(), labels.cuda()
        
        lateral_optimizer.zero_grad()
        output = lateral_model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        lateral_optimizer.step()
        
        lateral_losses[epoch] += loss.item()
    
    frontal_losses[epoch] /= len(frontal_train_dl.dataset)
    lateral_losses[epoch] /= len(lateral_train_dl.dataset)
    
    print('Train Losses: {} {}'
            .format(frontal_losses[epoch], lateral_losses[epoch]))
    
    if len(frontal_val_dl.dataset) > 0:
        frontal_model.eval()

        with torch.no_grad():
            for i, data in enumerate(frontal_val_dl):
                images, labels = data
                if cuda:
                    images, labels = images.cuda(), labels.cuda()

                output = frontal_model(images)

                loss = loss_fn(output, labels)

                frontal_validation[epoch] += loss.item()
        
        frontal_validation[epoch] /= len(frontal_val_dl.dataset)

    if len(lateral_val_dl.dataset) > 0:
        lateral_model.eval()

        with torch.no_grad():
            for i, data in enumerate(lateral_val_dl):
                images, labels = data
                if cuda:
                    images, labels = images.cuda(), labels.cuda()

                output = lateral_model(images)

                loss = loss_fn(output, labels)

                lateral_validation[epoch] += loss.item()

        lateral_validation[epoch] /= len(lateral_val_dl.dataset)

    print('Validation Losses: {} {}'
                    .format(frontal_validation[epoch],
                            lateral_validation[epoch]))

    frontal_scheduler.step()
    lateral_scheduler.step()

    print('Time to run: {}'.format(time.time()-start_time))

if not os.path.exists('/home/dqin/CS156b/'+name):
    os.makedirs('/home/dqin/CS156b/'+name)

torch.save(frontal_model.state_dict(),
           '/home/dqin/CS156b/'+
           name+
           '/frontal.pt'
)

torch.save(lateral_model.state_dict(),
           '/home/dqin/CS156b/'+
           name+
           '/lateral.pt'
)

test_dataloader = utils.gen_test_dataloader(path='test_ids.csv',
                                            batch_size=1,
                                            transform=transform
)

frontal_model.eval()
lateral_model.eval()

import pandas as pd

cols = [
    'ID', 
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]

out = []

start = time.time()

with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        num, xray_type, image = data
        image = image.cuda()

        if xray_type == 'frontal':
            p = frontal_model(image)
        else:
            p = lateral_model(image)

        p = p.flatten().tolist()

        out.append([num.item()] + p)

out = pd.DataFrame(out, columns=cols)

out.to_csv('out/'+name+'_out.csv', index=False)

print('Saved to file: '+name+'_out.csv')
print('Time to predict: {}'.format(time.time() - start))
