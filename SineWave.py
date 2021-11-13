# import  torchvision.transforms as transforms
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor
from torchmeta.utils.data import BatchMetaDataLoader
from    PIL import Image
import  os.path
import  numpy as np

import pdb


class SineWave:

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

        dataset = Sinusoid(num_samples_per_task=k_shot, num_tasks=35)

        self.dataloader = BatchMetaDataLoader(dataset, batch_size=batchsz, num_workers=4)

        dataset_val = Sinusoid(num_samples_per_task=k_shot, num_tasks=1000)

        self.dataloader_val = BatchMetaDataLoader(dataset_val, shuffle=True, batch_size=batchsz, num_workers=4)

