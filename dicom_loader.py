import os
import numpy as np
import torch
from torch.utils import data
from PIL import Image
import natsort
from pydicom import dcmread
import nibabel as nib
import glob
import matplotlib.pyplot as plt
class LungCancerDataset(data.Dataset):
    def __init__(self, root_path, size, mode="train"):
        super(LungCancerDataset, self).__init__()

        self.root_path = root_path
        self.mode = mode
        self.transforms = None
        self.dcm = None
        self.nifti = None

        if (self.mode == "train"):
            train_dir = os.path.join(self.root_path, "train")
            self.dcm = natsort.natsorted(glob.glob(os.path.join(train_dir,"**","*.dcm"),recursive=True))
            self.nifti = natsort.natsorted(glob.glob(os.path.join(train_dir,"**","*.nii"),recursive=True))
        elif (self.mode == "valid"):
            val_dir = os.path.join(self.root_path, "valid")
            self.dcm = natsort.natsorted(glob.glob(os.path.join(val_dir,"**","*.dcm"),recursive=True))
            self.nifti = natsort.natsorted(glob.glob(os.path.join(val_dir,"**","*.nii"),recursive=True))
        elif self.mode == "test":
            test_dir = os.path.join(self.root_path,"test")
            self.dcm = natsort.natsorted(glob.glob(os.path.join(test_dir,"**","*.dcm"),recursive=True))
            self.nifti = natsort.natsorted(glob.glob(os.path.join(test_dir,"**","*.nii"),recursive=True))
        else:
            raise NotImplementedError

    def __len__(self):
            return len(self.dcm)

    def __getitem__(self, idx):
        ds = dcmread(self.dcm[idx])
        dcm = (ds.pixel_array/4095).astype(np.float32)

		### display range apply 
		# center = ds[0x0028, 0x1050].value
		# width = ds[0x0028, 0x1051].value
		# minRange = int(center - round(width/2))
		# maxRange = int(center + int(width/2))

		# s = int(ds.RescaleSlope)
		# b = int(ds.RescaleIntercept)
		# im = s*ds.pixel_array+b
		
		# crit = -1*np.min(im)
		# im = im - np.min(im)

		# im = np.clip(im , np.min(im)+(crit+minRange),np.min(im)+(crit+maxRange))
		# image = (im/4095).astype(np.float32)
		###

        # fig = plt.figure()
        # ax1 = fig.add_subplot(1,2,1)
        # ax1.imshow(dcm,cmap='gray')

        dcm= torch.from_numpy(dcm).contiguous().type(torch.FloatTensor)
        dcm = dcm.unsqueeze(0)
        # print(dcm.shape)

        nifti = nib.load(self.nifti[idx])
        gt = nifti.get_fdata()
        # print(gt.shape)

        # ax2 = fig.add_subplot(1,2,2)
        # ax2.imshow(gt.transpose((1,0)),cmap='gray')

        # plt.show()
        gt = torch.from_numpy(gt.transpose((1,0))).type(torch.FloatTensor)

        sample = {"dcm": dcm, "nifti": gt, "affine":nifti.affine}

        return sample