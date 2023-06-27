我的雲端硬碟
import csv
import cv2
import numpy as np
import random
import os

from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models,transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import string
from accelerate import Accelerator
from accelerate import init_empty_weigh
from torcheval.metrics.functional import throughput
import time
accelerator = Accelerator()

Path3 = "task4.pt"
TRAIN_PATH = "train"
TEST_PATH = "test"
BATCH = 10
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = accelerator.device

# try device = "cuda" 
# and change your settings/accelerator to GPU if you want it to run faster
ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
NUM_ALPHA = NUMBER + ALPHABET
def encode(label):
    ohlabel = []
    for l in label:
        oh = [0]*len(NUM_ALPHA)
        idx = NUM_ALPHA.index(l)
        oh[idx] = 1
        ohlabel += oh
    #print(label)
    #print(ohlabel)
    return np.array(ohlabel)

class Task3Dataset(Dataset):
    def __init__(self, data, root, return_filename=False):
        self.data = [sample for sample in data if sample[0].startswith("task3")]
        self.return_filename = return_filename
        self.root = root
        self.captchalen = 4
        
    def __getitem__(self, index):
        filename, label = self.data[index]
        img = cv2.imread(f"{self.root}/{filename}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w= img.shape
        #print(h,w) 72 96
        #img = cv2.resize(img, (32, 32))
        #img = np.mean(img, axis=2)
        #imgplot = plt.imshow(img)
        #plt.show()
        if self.return_filename:
            return torch.FloatTensor(img), filename
        else:
            return torch.FloatTensor(img), encode(label)

    def __len__(self):
        return len(self.data)
    


class Model(nn.Module):
    def __init__(self, OUTPUT_LEN, TEMP_OUT):
        super().__init__()
        self.OUTPUT_LEN = OUTPUT_LEN
        self.TEMP_OUT = TEMP_OUT
        self.conv1 = nn.Sequential(
            #nn.Conv2d(1, 3, kernel_size=3),
            nn.Conv2d(1, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            
            nn.Conv2d(16, 128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=5),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # convolutional layer (sees 30*30*3 tensor)
        # linear layer (28*28*3 -> 100)
        self.fc1 = nn.Linear(TEMP_OUT, 500)
        self.drop = nn.Dropout(0.2)
        # linear layer (100 -> 10)
        self.fc2 = nn.Linear(500, self.OUTPUT_LEN)
        
        
    def forward(self, x):
        #print(x.size())
        batch, height, width = x.shape
        x = x.view(batch, 1, height, width)
        #print(x.size())
        # sequance of convolutional layers with relu activation
        x = self.conv1(x)
        #print(x.size())
        x = self.conv2(x)
        #x = self.drop(x)
        #print(x.size())
        x = self.conv3(x)
        x = self.drop(x)
        #print(x.size())
        #x = self.conv4(x)
        # flatten the image input
        #print(x.shape)
        x = x.view(-1, self.TEMP_OUT)
        # 1st hidden layer with relu activation
        #print(x.size())
        x = F.relu(self.fc1(x))
        # output-layer
        #print(x.size())
        
        #print(x.size())
        x = self.fc2(x)
        #print(x.size())
        return x
    
test_data = []
with open(f'submission.csv', newline='') as csvfile:
    for row in csv.reader(csvfile, delimiter=','):
        test_data.append(row)
test3_ds = Task3Dataset(test_data, root=TEST_PATH, return_filename=True)
test3_dl = DataLoader(test3_ds,batch_size=BATCH, num_workers=0, drop_last=False, shuffle=False)


def test(test_ds, test_dl, OUTPUT_LEN, PATH, TEMP_OUT = 4096):
    task = test_ds.captchalen
    csv_writer = csv.writer(open('submission.csv', 'a', newline=''))

    #load model
    model = Model(OUTPUT_LEN=OUTPUT_LEN,TEMP_OUT=TEMP_OUT).to(device)
    # ADDED
    model, test_dl = accelerator.prepare(model, test_dl)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(epoch)
    
    model.eval()
    sum_tensor = torch.tensor(0.0)
    for image, filenames in test_dl:
        idx=0
        image = image.to(device)
        start_time = time.time()
        pred = model(image)
        elapsed_time = time.time()-start_time
        pred_str = [""]*len(filenames)
        thru = throughput(len(filenames), elapsed_time)
        print('throughput: ', thru, len(filenames))
        sum_tensor += thru
        '''for i in range(task):
            pred_i = torch.argmax(pred[:,i*36:(i+1)*36], dim=1)
            for b in range(BATCH):
                pred_str[b] += NUM_ALPHA[pred_i[b]]'''

        '''for i in range(len(filenames)):
            csv_writer.writerow([filenames[i], pred_str[i]])'''
    print('sum: ', torch.mean(sum_tensor))

test(test3_ds, test3_dl, 144, Path3, TEMP_OUT=7168)
