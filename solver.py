import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from network import U_Net
from PIL import Image
from tqdm import tqdm
import nibabel as nib
import pydicom
from evaluation import eval_volume_from_mask
class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()

    def forward(self, gt, predict, smooth=1e-5):
        intersection = (gt * predict).sum()
        dice_coefficient = (2*abs(intersection) + smooth) / (abs(gt.sum()) + abs(predict.sum()) + smooth)

        loss_dice = (1-dice_coefficient)

        return loss_dice

class Solver(object):
	def __init__(self, config):
		# Models
		self.model = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = DiceLoss()

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode
		self.pretrain_model = config.pretrain_model

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.test_model = config.test_model
		self.build_model()

	def build_model(self):
		self.model = U_Net(img_ch=self.img_ch,output_ch=1)
		self.optimizer = optim.Adam(list(self.model.parameters()),
									  self.lr, [self.beta1, self.beta2],eps =1e-8)
		self.model.to(self.device)

		# self.print_network(self.unet, self.model_type)

	def train(self,train_loader, valid_loader):

		#====================================== Training ===========================================#
		#===========================================================================================#
		lr = self.lr
		min_loss = 99999
		for epoch in range(self.num_epochs):
			self.model.train(True)
			train_losses = []
			valid_losses = []
			for i, data in enumerate(tqdm(train_loader)):
				dcm = data["dcm"].to(self.device)
				gt = data["nifti"].to(self.device)

				pred= self.model(dcm)
				pred = torch.sigmoid(pred)
				loss = self.criterion(gt,pred)
				train_losses.append(loss.item())

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

			self.model.train(False)
			self.model.eval()
		#====================================== validation ===========================================#
		#===========================================================================================#
			for i, data in enumerate(tqdm(valid_loader)):
				dcm = data["dcm"].to(self.device)
				gt = data["nifti"].to(self.device)

				pred= self.model(dcm)
				pred = torch.sigmoid(pred)
				self.criterion=DiceLoss() #BCE not use
				loss = self.criterion(gt,pred)
				valid_losses.append(loss.item())
				
			# Print the log info
			train_loss = np.average(train_losses)
			valid_loss = np.average(valid_losses)
			print('Epoch [%d/%d], Train Loss: %.4f,  Valid Loss: %.4f, LR:%f' % 
								(epoch+1, self.num_epochs, train_loss, valid_loss,lr))
			unet_path = os.path.join(self.model_path, '%s_%d_%f.pkl' % ("unet",epoch+1,lr))

			if min_loss > valid_loss : 
				min_loss = valid_loss
				torch.save(self.model.state_dict(), unet_path)

	def test(self, test_loader):
		model_path = os.path.join(self.model_path,self.pretrain_model)
		self.model.load_state_dict(torch.load(model_path))
		self.model.train(False)
		self.model.eval()

		for i, data in enumerate(tqdm(test_loader)):
			dcm = data["dcm"].to(self.device)
			# print(raw_dcm.shape)
			gt = data["nifti"]
			# print("gt ",gt.shape)
			affine = data["affine"].squeeze(0).numpy()
			nifti_file_name = data["nifti_file_name"]
			# print(file_name)

			pred = torch.sigmoid(self.model(dcm))
			pred = (pred>0.5).float()
			pred0 = pred.squeeze(0).cpu()
			# print("float shape", pred0.shape)
			
			gt = nib.Nifti1Image(gt.numpy().transpose((2,1,0)),affine=affine)
			pred = nib.Nifti1Image(pred0.numpy().transpose((2,1,0)),affine=affine)
			nib.save(gt, os.path.join(self.result_path,"GT",nifti_file_name[0]))
			nib.save(pred, os.path.join(self.result_path,"Predict",nifti_file_name[0]))
		
		eval_volume_from_mask(os.path.join(self.result_path,"GT"),os.path.join(self.result_path,"Predict"))
