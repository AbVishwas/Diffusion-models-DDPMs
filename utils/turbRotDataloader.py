from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Lambda
from PIL import Image
import h5py
import numpy as np

class turbRotDataset(Dataset):
    """
    Custom dataset class for turbrot dataset. it will help dataloader to find the length of dataset (__len__)
    and access each image one by one (__getitem__)
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = Compose([                                               #Compose class is typically used to define pipelines to perfrom data processing (mostly for images)
                            ToTensor(),                                          #ToTensor() convert image to pytorch tensor in (Channel, Height, width)
                            Lambda(lambda x: (x - 0.5)*2)                        #Custom lambda function: substracts 0.5 from each element and multiplies by 2. To normalize the data between [-1,1]
                        ])

        with h5py.File(self.data_dir, 'r') as hf:
            self.flowMean = np.array(hf.get('Mean'))
            self.Semidisp = np.array(hf.get('Semidisp'))
            self.train    = np.array(hf.get('train'))
            self.test     = np.array(hf.get('test'))
            self.val      = np.array(hf.get('dev'))

    # Initialize your data or provide a list of file paths.    
    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        # Load and preprocess your data here
        image = self.transform(self.train[idx])                                            #Note that we are applying transform here

        return image