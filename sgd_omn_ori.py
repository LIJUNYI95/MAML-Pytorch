import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmeta.datasets import Omniglot, MiniImagenet
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor
import torchvision.models as models
from torchmeta.utils.data import BatchMetaDataLoader
import os
import time
import pdb
from my_optimizer import Adam
from collections import defaultdict




os.environ["CUDA_VISIBLE_DEVICES"] = "1"
log_file = open("log_ori_3.txt","a")

innerT = 10
innerT_test = 10
T = 2001
hlr = 0.001
weight_decay = 1e-5
lr = 0.1
gpu_num = 0
batch_size_out = 4
num_class = 5
input_channels = 1

np.random.seed(19260817)
torch.manual_seed(19260817)

def convLayer(in_planes, out_planes, useDropout = False):
    "3x3 convolution with padding"
    seq = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    return seq

class META(nn.Module):
    def __init__(self):
        super(META, self).__init__()
        self.layer1 = convLayer(input_channels,64)
        self.layer2 = convLayer(64,64)
        self.layer3 = convLayer(64,64)
        self.layer4 = convLayer(64,64)

        self.weights_init(self.layer1)
        self.weights_init(self.layer2)
        self.weights_init(self.layer3)
        self.weights_init(self.layer4)

   
    def weights_init(self,module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_norm(self, module):
        norm = 0
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                norm += torch.norm(m.weight.grad).data.cpu().numpy()
                norm += torch.norm(m.bias.grad).data.cpu().numpy()
        return norm

    def get_grad_norm(self):
        return self.get_norm(self.layer1) + self.get_norm(self.layer2) + \
            self.get_norm(self.layer3) + self.get_norm(self.layer4)


    def forward(self, image_input):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :return: Embeddings of size [batch_size, 64]
        """
        #pdb.set_trace()
        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        return x


class META_RES(nn.Module):
    def __init__(self):
        super(META_RES, self).__init__()
        self.res = models.resnet18()

    def get_norm(self, module):
        norm = 0
        for m in module.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                norm += torch.norm(m.weight.grad).data.cpu().numpy()
                if m.bias is not None:
                    norm += torch.norm(m.bias.grad).data.cpu().numpy()              
        return norm

    def get_grad_norm(self):
        norm = 0
        for module in self.res.modules():
            norm += self.get_norm(module)
        return norm


    def forward(self, image_input):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :return: Embeddings of size [batch_size, 64]
        """
        #pdb.set_trace()
        return self.res(image_input)


class INNER(nn.Module):
    def __init__(self):
        super(INNER, self).__init__()
        self.meta = META()
        self.weight = torch.tensor(np.random.normal(size = (batch_size_out, 64, num_class)),\
            dtype=torch.float).cuda().requires_grad_()
        self.bias = torch.tensor(np.random.normal(size = (batch_size_out, 1, num_class)),\
            dtype=torch.float).cuda().requires_grad_()

    def forward(self,x):
        #pdb.set_trace()
        batch_outer, batch_inner = x.shape[0:2]
        x = torch.reshape(x,(batch_outer * batch_inner,input_channels,28,28))
        x = self.meta(x)
        x = torch.reshape(x,(batch_outer, batch_inner,-1))
        #pdb.set_trace() 
        x = torch.matmul(x, self.weight)
        x = torch.add(x, self.bias)
        return x

class GHO(nn.Module):
    def __init__(self):
        super(GHO, self).__init__()
        self.inner = INNER()
   
    def forward(self,data):
        x,y = data[0], data[1]
        x = self.inner(x)
        #pdb.set_trace()
        loss =  F.cross_entropy(x.reshape(-1,num_class), y.reshape(-1))
        return loss


dataset = Omniglot("data",
                   # Number of ways
                   num_classes_per_task=num_class,
                   # Resize the images to 28x28 and converts them\
                   #  to PyTorch tensors (from Torchvision)
                   transform=Compose([Resize(28), ToTensor()]),
                   # Transform the labels to integers (e.g.\
                   #  ("Glagolitic/character01", "Sanskrit/character14", ...) \
                   # to (0, 1, ...))
                   target_transform=Categorical(num_classes=num_class),
                   # Creates new virtual classes with rotated versions \
                   # of the images (from Santoro et al., 2016)
                   class_augmentations=[Rotation([90, 180, 270])],
                   meta_train=True,
                   download=True)

dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=1, \
    num_test_per_class=15)
dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size_out, num_workers=4)

dataset_val = Omniglot("data",
                   # Number of ways
                   num_classes_per_task=num_class,
                   # Resize the images to 28x28 and converts them\
                   #  to PyTorch tensors (from Torchvision)
                   transform=Compose([Resize(28), ToTensor()]),
                   # Transform the labels to integers (e.g.\
                   #  ("Glagolitic/character01", "Sanskrit/character14", ...) \
                   # to (0, 1, ...))
                   target_transform=Categorical(num_classes=num_class),
                   # Creates new virtual classes with rotated versions \
                   # of the images (from Santoro et al., 2016)
                   class_augmentations=[Rotation([90, 180, 270])],
                   meta_val=True,
                   download=True)

dataset_val = ClassSplitter(dataset_val, shuffle=True, num_train_per_class=1, \
    num_test_per_class=15)
dataloader_val = BatchMetaDataLoader(dataset_val, shuffle=True, batch_size=batch_size_out, num_workers=4)

now = time.time()
# with torch.cuda.device(gpu_num):

gho = GHO()
gho = gho.cuda()

optimizer_inner = Adam(['w','b'], lr=lr)
optimizer_outer = optim.Adam(gho.inner.meta.parameters(),\
    lr=hlr, weight_decay=weight_decay)
hyper_grad_norm = []
val_err = []
accuracy = []

for hyt, data in enumerate(dataloader):
    if hyt > T:
        break
    #evaluation
    if hyt % 25 == 0:
        acc_inner = []
        for hyt_val, data_val in enumerate(dataloader_val):
            if hyt_val > 100:
                break
            optimizer_inner.state = defaultdict(dict)
            
            nn.init.xavier_uniform_(gho.inner.weight)
            nn.init.uniform_(gho.inner.bias)
            gho.inner.weight.detach_().requires_grad_()
            gho.inner.bias.detach_().requires_grad_()

            #pdb.set_trace()
            x, y = data_val["train"]
            # x = torch.squeeze(x,dim=0).cuda()
            # y = torch.squeeze(y).cuda()
            dX, dY = data_val["test"]
            index = np.zeros(num_class*15,np.bool_)
            val_index = np.random.choice(num_class*15,size=10*num_class, replace=False)
            index[val_index] = np.ones(num_class*10,np.bool_)
            devX, testX = dX[:,index], dX[:,~index]
            devY, testY = dY[:,index], dY[:,~index]
            x = torch.cat((x,devX),dim=1)
            y = torch.cat((y,devY),dim=1)
            # devX = torch.squeeze(devX,dim=0).cuda()
            # devY = torch.squeeze(devY).cuda()
            x, y = x.cuda(), y.cuda()
            devX, devY = devX.cuda(), devY.cuda()
            testX, testY = testX.cuda(), testY.cuda()
            #pdb.set_trace()

            new1 = time.time()
            for epoch in range(innerT_test):
                loss = gho((x,y))

                new_params = optimizer_inner.step(loss = loss,\
                    weights = [gho.inner.weight,gho.inner.bias], \
                        create_graph=True)
                gho.inner.weight =gho.inner.weight + new_params[0]
                gho.inner.bias =gho.inner.bias + new_params[1]

            ans = torch.argmax(gho.inner((testX)).reshape(-1,num_class),-1)
            acc = torch.mean(ans == testY.reshape(-1),dtype = torch.float)
            acc_inner.append(acc.data.cpu().numpy())

        accuracy.append(np.mean(acc_inner))
        print("the accuracy is:", accuracy[-1])

    optimizer_inner.state = defaultdict(dict)
    
    nn.init.xavier_uniform_(gho.inner.weight)
    nn.init.uniform_(gho.inner.bias)
    gho.inner.weight.detach_().requires_grad_()
    gho.inner.bias.detach_().requires_grad_()


    #pdb.set_trace()
    x, y = data["train"]
    x, y = x.cuda(), y.cuda()
    dX, dY = data["test"]
    index = np.zeros(num_class*15,np.bool_)
    val_index = np.random.choice(num_class*15,size=10*num_class,replace=False)
    index[val_index] = np.ones(num_class*10,np.bool_)
    devX, testX = dX[:,index], dX[:,~index]
    devY, testY = dY[:,index], dY[:,~index]
    devX, devY = devX.cuda(), devY.cuda()
    testX, testY = testX.cuda(), testY.cuda()
    #pdb.set_trace()

    new1 = time.time()
    for epoch in range(innerT):
        loss = gho((x,y))

        print("The training loss at iter", epoch, "is: ", loss.data.cpu().numpy())
        new_params = optimizer_inner.step(loss = loss,\
                weights = [gho.inner.weight,gho.inner.bias], \
                    create_graph=True)
        gho.inner.weight =gho.inner.weight + new_params[0]
        gho.inner.bias =gho.inner.bias + new_params[1]
    print("training time is: ", time.time() - new1)

    now1 = time.time()
    ERR = gho((devX,devY))
    error = torch.norm(ERR).data.cpu().numpy()
    val_err.append(error)
    print("The validation Err is:", val_err[-1])
    optimizer_outer.zero_grad()
    ERR.backward()
    optimizer_outer.step()
    norm = gho.inner.meta.get_grad_norm()
    hyper_grad_norm.append(norm)
    print("The norm of the hypergradient is: ", norm)


    if hyt%1 == 0:
        print(hyt,':',ERR.data.cpu().numpy(),file=log_file)#,acc.data.cpu().numpy())
    if hyt%200==0:
        np.save("grad_norm/meta_sgd_omni_grad_norm_1_5_4_10", hyper_grad_norm) 
        np.save("grad_norm/meta_sgd_omni_val_err_1_5_4_10", val_err)
        np.save("grad_norm/meta_sgd_omni_acc_1_5_4_10", accuracy) 
print("running time is: %.3f", time.time() - now)
   