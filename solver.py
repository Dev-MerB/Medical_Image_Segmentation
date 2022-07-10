import os
import numpy as np
import torch
import torchvision
from torch import optim
import torch.nn.functional as F
from skimage.transform import resize
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net,Nested_UNet
import csv
from PIL import Image
import cv2
from vol_eval import *
from tqdm import tqdm
from pytorchtools import EarlyStopping
import nibabel as nib
import matplotlib.pyplot as plt
class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()

    def forward(self, gt, predict, smooth=1e-5):

        # gt = one_hot encoding
        # predict = network output
        intersection = (gt * predict).sum()
        dice_coefficient = (2*abs(intersection) + smooth) / (abs(gt.sum()) + abs(predict.sum()) + smooth)
        flat_gt = gt.view(gt.size(0),-1)
        flat_pred = predict.view(predict.size(0),-1)

        loss_dice = (1-dice_coefficient)
        loss_func =  torch.nn.BCELoss()
        loss_bce = loss_func(flat_pred,flat_gt)
        loss_total = 0.5*loss_bce+loss_dice
        # loss_total = torch.log((torch.exp(loss_dice) + torch.exp(-loss_dice)) / 2.0)

        return loss_total

class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = torch.nn.BCELoss()
		self.augmentation_prob = config.augmentation_prob

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t

		self.test_model = config.test_model
		self.train_path = config.train_path
		self.test_path = config.test_path
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch=self.img_ch,output_ch=1)
		elif self.model_type =='R2U_Net':
			self.unet = R2U_Net(img_ch=self.img_ch,output_ch=1,t=self.t)
		elif self.model_type =='AttU_Net':
			self.unet = AttU_Net(img_ch=self.img_ch,output_ch=1)
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(img_ch=self.img_ch,output_ch=1,t=self.t)
		elif self.model_type == 'NestedU_Net':
			self.unet = Nested_UNet(img_ch=self.img_ch,output_ch=1)

		self.optimizer = optim.Adam(list(self.unet.parameters()),
									  self.lr, [self.beta1, self.beta2],eps =1e-8)
		self.unet.to(self.device)

		# self.print_network(self.unet, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)

		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img


	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#
		
		#unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))
		unet_path = os.path.join(self.model_path, 's.pkl')
		early_stopping = EarlyStopping(patience=250)
		train_losses = []
		valid_losses = []
		# U-Net Train
		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
		else:
			# Train for Encoder
			lr = self.lr
			min_loss = 99999
			for epoch in range(self.num_epochs):

				self.unet.train(True)
				epoch_loss = 0
				for i, (images, GT,empty) in enumerate(tqdm(self.train_loader)):
					# GT : Ground Truth

					images = images.to(self.device)
					GT = GT.to(self.device)
					
					# SR : Segmentation Result
					SR = self.unet(images)
					SR_probs = torch.sigmoid(SR)

					# SR_flat = SR_probs.view(SR_probs.size(0),-1)

					# GT_flat = GT.view(GT.size(0),-1)
					# loss =self.criterion(SR_flat,GT_flat)

					self.criterion=DiceLoss() #BCE not use
					loss = self.criterion(GT,SR_probs)
					train_losses.append(loss.item())
					# Backprop + optimize
					self.reset_grad()
					loss.backward()
					self.optimizer.step()

				if(epoch+1) >(self.num_epochs- self.num_epochs_decay):
					lr -= (self.lr/float(self.num_epochs_decay))
					for param_group in self.optimizer.param_groups:
						param_group['lr']=lr
					print('Decay learning rate to lr :{}'.format(lr))
				######################
				# Validate the model #
				######################
				self.unet.train(False)
				self.unet.eval()
				for i, (images,GT,empty) in enumerate(tqdm(self.valid_loader)):
					images = images.to(self.device)
					GT = GT.to(self.device)
					SR = torch.sigmoid(self.unet(images))

					# SR_flat = SR.view(SR.size(0),-1)

					# GT_flat = GT.view(GT.size(0),-1)
					# loss = self.criterion(SR_flat,GT_flat)
					loss = self.criterion(GT,SR)
					epoch_loss += loss.item()

					valid_losses.append(loss.item())

				# Print the log info
				train_loss = np.average(train_losses)
				valid_loss = np.average(valid_losses)
				valid_losses = []
				train_losses = []
				print('Epoch [%d/%d], Train Loss: %.4f,  Valid Loss: %.4f, LR:%f' % 
									(epoch+1, self.num_epochs, train_loss, valid_loss,lr))
				
				early_stopping(valid_loss, self.unet)
				
				unet_path = os.path.join(self.model_path,"0215_EV3", '%s_%d_%f.pkl' % (self.model_type,epoch+1,lr))

				if min_loss > epoch_loss : 
					min_loss = epoch_loss
					torch.save(self.unet.state_dict(), unet_path)

				if early_stopping.early_stop:
					print("Stopped ", epoch+1)
					torch.save(self.unet.state_dict(), unet_path)
					break

	def test(self):

		del self.unet
		self.build_model()
		# unet_path = os.path.join(self.model_path, '%s-%d-%d.pkl' % (self.model_type, self.fold, epoch))
		unet_path = os.path.join(self.model_path,'%s.pkl' % (self.test_model))
		print(unet_path)
		self.unet.load_state_dict(torch.load(unet_path))

		self.unet.train(False)
		self.unet.eval()

		for i, (images, GT,image_path) in enumerate(tqdm(self.test_loader)):
			images= images.to(self.device)
			image_path = list(image_path)
			gt = GT[0][0].cpu().numpy()
			gt = gt.transpose((1,0)).astype(np.uint16)
			

			SR = torch.sigmoid(self.unet(images))
			SR = (SR>0.5).float()
			SR = SR[0][0].cpu().numpy()
			SR = SR.transpose((1,0)).astype(np.uint16)

			
			pred = nib.Nifti1Image(SR,affine=np.eye(4))
			gt_o = nib.Nifti1Image(gt,affine=np.eye(4))
			
			# fig = plt.figure()		
			# ax1 = fig.add_subplot(1,3,1)
			# ax1.imshow(SR,cmap='gray')
			# ax2 = fig.add_subplot(1,3,2)
			# ax2.imshow(images.cpu().numpy().squeeze(), cmap='gray')
			# ax3 = fig.add_subplot(1,3,3)
			# ax3.imshow(gt,cmap='gray')
			# plt.show()

			fx = 320 / 256
			fy = 280 / 256
			rescale = resize(SR,(320,280,1))
			rescale = nib.Nifti1Image(rescale,affine=np.eye(4))

			nib.save(gt_o, os.path.join(self.result_path,"GT",image_path[0].split(".")[0]+".nii"))
			nib.save(pred, os.path.join(self.result_path,"Predict",image_path[0].split(".")[0]+".nii"))
			nib.save(rescale, os.path.join(self.result_path,"Rescale",image_path[0].split(".")[0]+".nii"))


		eval_volume_from_mask( os.path.join(self.result_path,"GT"), os.path.join(self.result_path,"Predict"))
		