import os
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import natsort
from pydicom import dcmread
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
# np.set_printoptions(threshold = np.inf, linewidth = np.inf)
class ImageFolder(data.Dataset):
	def __init__(self, root,mode='train',augmentation_prob=0.4, model='U_Net'):
		"""Initializes image paths and preprocessing module."""
		self.root = root
		
		# GT : Ground Truth
		self.model = model
		if model == 'MIA':
			self.GT_paths = root[:-1]+'_MIA_GT/'
		else:
			self.GT_paths = root[:-1]+'_GT/'
		self.image_paths = list(natsort.natsorted(os.listdir(root)))
		self.mode = mode
		self.augmentation_prob = augmentation_prob
		print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

	def __getitem__(self, index):
		image_path = self.image_paths[index]
		image_GT_path = image_path[:8]+'_'+image_path[8:12]+'.nii'
		GT_path = self.GT_paths + image_GT_path

		ds = dcmread(self.root+image_path)
		image = (ds.pixel_array/4095).astype(np.float32)

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
		# fig = plt.figure()
		# ax1 = fig.add_subplot(1,2,1)
		# ax1.imshow(image,cmap='gray')
		
		image= torch.from_numpy(image).contiguous().type(torch.FloatTensor)
		
		image = image.unsqueeze(0)
		# print(image.shape)


		gt = nib.load(GT_path)
		gt = gt.get_fdata()
		# ax2 = fig.add_subplot(1,2,2)
		# ax2.imshow(gt.transpose((1,0,2)),cmap='gray')
		# plt.show()
		gt = torch.from_numpy(gt.transpose((2,1,0))).type(torch.FloatTensor)
		# print("gt :", gt.shape)
		return image, gt, image_path

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.root+str(self.image_paths))

def get_dicom_loader(image_path, batch_size, num_workers=2, mode='train',augmentation_prob=0.4,model='U-Net'):
	"""Builds and returns Dataloader."""

	data_image = ImageFolder(root = image_path, mode=mode,augmentation_prob=augmentation_prob,model = model)
	#test_index = []

	# index = 0
	# for path in dataset.image_paths:
	# 	split_path = path.split("_")
	# 	if int(split_path[0]) in test_idx:
	# 		test_index.append(index)

	# 	index = index + 1

	# total_index = list(range(0, len(dataset.image_paths)))


	# train_idx = [index for index in total_index if index not in test_index]

	# dataset_train = torch.utils.data.Subset(dataset, train_idx)
	# dataset_test = torch.utils.data.Subset(dataset, test_index)

	dataset = torch.utils.data.Subset(data_image, list(range(0, len(data_image.image_paths))))
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)

	return data_loader