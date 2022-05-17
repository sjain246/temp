import torch.nn as nn
import numpy as np
import torch.optim as optim
import utils
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.models import densenet121
import os

import time

name = 'densenet_indv_all'

all_data = True
n_examples = 7500

n_split = 0

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

transform = transforms.Compose([
                     transforms.Resize((224, 224)),
                     transforms.Normalize([123.1379],
                                          [77.6652]),
                     transforms.Lambda(
                        lambda x: x.repeat(3,1,1)
                     )
                 ])

cuda = torch.cuda.is_available()

print(cuda)
print(torch.cuda.device_count())

cats = [
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

class DenseNet(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(DenseNet,self).__init__()
        
        # If we want to do the 5 channel crop thing, here is where to add the layer
        # self.c1 = nn.Conv2d(5, 3, kernel_size=3)
        self.densenet121 = densenet121(pretrained=True)       
        num_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # x = F.leaky_relu(F.max_pool2d(self.c1(x),2))
        x = self.densenet121(x)
        return x

frontal_datasets = {}

lateral_datasets = {}

for cat in cats:
    if all_data:
        sample = 'all'
    else:
        sample = 'random'
        
    frontal_datasets[cat] = utils.CheXDataset(csv_file='train.csv',
                                             data_dir='/groups/CS156b/data/',
                                             n_samples=n_examples,
                                             sample_type=sample,
                                             xray_type='frontal',
                                             pathology=cat,
                                             fill_type='exclude',
                                             transform=transform)
    
    lateral_datasets[cat] = utils.CheXDataset(csv_file='train.csv',
                                             data_dir='/groups/CS156b/data/',
                                             n_samples=n_examples,
                                             sample_type=sample,
                                             xray_type='lateral',
                                             pathology=cat,
                                             fill_type='exclude',
                                             transform=transform)
    
    print(cat)
    print(len(frontal_datasets[cat]))
    print(len(lateral_datasets[cat]))

frontal_train = {}
frontal_val = {}

lateral_train = {}
lateral_val = {}

for cat in cats:
    n_train = int((1-n_split)*len(frontal_datasets[cat]))
    n_val = len(frontal_datasets[cat]) - n_train
    
    frontal_train[cat], frontal_val[cat] = random_split(frontal_datasets[cat],
                                                        [n_train, n_val])
    
    n_train = int((1-n_split)*len(lateral_datasets[cat]))
    n_val = len(lateral_datasets[cat]) - n_train
    
    lateral_train[cat], lateral_val[cat] = random_split(lateral_datasets[cat],
                                                        [n_train, n_val])
    
    print(cat)
    print(len(frontal_train[cat]), len(frontal_val[cat]))
    print(len(lateral_train[cat]), len(lateral_val[cat]))

frontal_train_dl = {}
frontal_val_dl = {}

lateral_train_dl = {}
lateral_val_dl = {}

for cat in cats:
    frontal_train_dl[cat] = DataLoader(frontal_train[cat],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True
                                 )

    lateral_train_dl[cat] = DataLoader(lateral_train[cat],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True
                                 )

    frontal_val_dl[cat] = DataLoader(frontal_val[cat],
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True
                               )

    lateral_val_dl[cat] = DataLoader(lateral_val[cat],
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True
                               )

frontal_models = {}
lateral_models = {}

frontal_optimizers = {}
lateral_optimizers = {}

frontal_schedulers = {}
lateral_schedulers = {}

for cat in cats:
    frontal_models[cat] = DenseNet()
    lateral_models[cat] = DenseNet()
    
    frontal_models[cat] = nn.DataParallel(frontal_models[cat])
    lateral_models[cat] = nn.DataParallel(lateral_models[cat])
    
    if cuda:
        frontal_models[cat].cuda()
        lateral_models[cat].cuda()
    
    frontal_optimizers[cat] = optim.Adam(frontal_models[cat].parameters(),
                                         lr=learning_rate,
                                         weight_decay=weight_decay)
    
    lateral_optimizers[cat] = optim.Adam(lateral_models[cat].parameters(),
                                         lr=learning_rate,
                                         weight_decay=weight_decay)
    
    frontal_schedulers[cat] = optim.lr_scheduler.StepLR(frontal_optimizers[cat],
                                                        step_size=step_size,
                                                        gamma=gamma)
    
    lateral_schedulers[cat] = optim.lr_scheduler.StepLR(lateral_optimizers[cat],
                                                        step_size=step_size,
                                                        gamma=gamma)

loss_fn = nn.MSELoss(reduction='sum')

frontal_losses = {}
lateral_losses = {}

frontal_validation = {}
lateral_validation = {}

for cat in cats:
    frontal_losses[cat] = np.zeros(n_epochs)
    lateral_losses[cat] = np.zeros(n_epochs)
    
    frontal_validation[cat] = np.zeros(n_epochs)
    lateral_validation[cat] = np.zeros(n_epochs)

for epoch in range(n_epochs):
    print('='*25)
    print('Epoch {}/{}:'.format(epoch+1, n_epochs))
    print('='*25)
    
    for cat in cats:
        start_time = time.time()
        
        print('-'*25)
        print(cat)
 
        
        frontal_models[cat].train()
        
        for i, data in enumerate(frontal_train_dl[cat]):
            images, labels = data
            if cuda:
                images, labels = images.cuda(), labels.cuda()
                
            frontal_optimizers[cat].zero_grad()
            output = frontal_models[cat](images)
            loss = loss_fn(output, labels)
                
            loss.backward()
            frontal_optimizers[cat].step()
            
            frontal_losses[cat][epoch] += loss.item()
        
        lateral_models[cat].train()
        
        for i, data in enumerate(lateral_train_dl[cat]):
            images, labels = data
            if cuda:
                images, labels = images.cuda(), labels.cuda()
            
            lateral_optimizers[cat].zero_grad()
            output = lateral_models[cat](images)
            loss = loss_fn(output, labels)
            loss.backward()
            lateral_optimizers[cat].step()
            
            lateral_losses[cat][epoch] += loss.item()
        
        frontal_losses[cat][epoch] /= len(frontal_train_dl[cat].dataset)
        lateral_losses[cat][epoch] /= len(lateral_train_dl[cat].dataset)
        
        print('Train Losses: {} {}'
                .format(frontal_losses[cat][epoch], lateral_losses[cat][epoch]))
        
        if len(frontal_val_dl[cat].dataset) > 0:
            frontal_models[cat].eval()

            with torch.no_grad():
                for i, data in enumerate(frontal_val_dl[cat]):
                    images, labels = data
                    if cuda:
                        images, labels = images.cuda(), labels.cuda()

                    output = frontal_models[cat](images)

                    loss = loss_fn(output, labels)

                    frontal_validation[cat][epoch] += loss.item()
            
            frontal_validation[cat][epoch] /= len(frontal_val_dl[cat].dataset)
 
        if len(lateral_val_dl[cat].dataset) > 0:
            lateral_models[cat].eval()

            with torch.no_grad():
                for i, data in enumerate(lateral_val_dl[cat]):
                    images, labels = data
                    if cuda:
                        images, labels = images.cuda(), labels.cuda()

                    output = lateral_models[cat](images)

                    loss = loss_fn(output, labels)

                    lateral_validation[cat][epoch] += loss.item()

            lateral_validation[cat][epoch] /= len(lateral_val_dl[cat].dataset)

        print('Validation Losses: {} {}'
                     .format(frontal_validation[cat][epoch],
                             lateral_validation[cat][epoch]))

        frontal_schedulers[cat].step()
        lateral_schedulers[cat].step()

        print('Time to run: {}'.format(time.time()-start_time))

if not os.path.exists('/home/dqin/CS156b/'+name):
    os.makedirs('/home/dqin/CS156b/'+name)

for cat in cats:
    torch.save(frontal_models[cat].state_dict(),
               '/home/dqin/CS156b/'+
               name+
               '/frontal_'
               +cat.lower().replace(' ', '_')
               +'.pt')
    
    torch.save(lateral_models[cat].state_dict(),
               '/home/dqin/CS156b/'+
               name+
               '/lateral_'
               +cat.lower().replace(' ', '_')
               +'.pt')
    

test_dataloader = utils.gen_test_dataloader(path='test_ids.csv',
                                            batch_size=1,
                                            transform=transform)

for cat in cats:
    frontal_models[cat].eval()
    lateral_models[cat].eval()

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
        
        preds = []

        for cat in cats:
            if xray_type == 'frontal':
                p = frontal_models[cat](image)
            else:
                p = lateral_models[cat](image)

            p = p.item()

            preds.append(p)

        out.append([num.item()] + preds)

out = pd.DataFrame(out, columns=cols)

out.to_csv('out/'+name+'_out.csv', index=False)

print('Saved to file: '+name+'_out.csv')
print('Time to predict: {}'.format(time.time() - start))
