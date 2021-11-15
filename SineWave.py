# import  torchvision.transforms as transforms
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor
from torchmeta.utils.data import BatchMetaDataLoader
from    PIL import Image
import  os.path
import  numpy as np
import torch
import pdb


class SineWave(object):

    def __init__(self, root, batchsz, k_shot):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """

        # dataset = Sinusoid(num_samples_per_task=k_shot + 15, num_tasks=100)

        # self.dataloader = BatchMetaDataLoader(dataset, batch_size=batchsz, num_workers=4)

        # dataset_val = Sinusoid(num_samples_per_task=k_shot + 15, num_tasks=100)

        # self.dataloader_val = BatchMetaDataLoader(dataset_val, shuffle=True, batch_size=batchsz, num_workers=4)

        self.k_shot = k_shot

        self.k_qry = 10

        self.batchsz = batchsz
    
    def gen_one_task(self):

        data_x = np.zeros((self.batchsz, self.k_shot + self.k_qry,1))
        data_y = np.zeros((self.batchsz, self.k_shot + self.k_qry,1))
        for i in range(self.batchsz):
            # A = np.random.randint(1, high=6)
            # Phi = np.random.randint(1, high=6)
            A = 1; Phi = 1

            data_x[i] = np.random.uniform(low=-5, high=5, size=self.k_shot + self.k_qry).reshape(1,-1,1)
            data_y[i] = A * np.sin(data_x[i] + Phi * np.pi/5)

        return torch.tensor(data_x).cuda(), torch.tensor(data_y).cuda()

    def gen_one_test_task(self):

        A = np.random.uniform(low = 1, high=5)
        Phi = np.random.uniform(low = 1, high=5)

        x = np.random.uniform(low=-5, high=5, size=self.k_shot + self.k_qry).reshape(1,-1,1)
        y = A * np.sin(x + Phi * np.pi/5)

        return torch.tensor(x).cuda(), torch.tensor(y).cuda()