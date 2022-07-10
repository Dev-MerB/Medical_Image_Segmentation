import argparse
import os
from solver import Solver
from dicom_loader import LungCancerDataset
import torch.utils.data as data
import random

def main(config):

   # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.test_model)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    if not os.path.exists(os.path.join(config.result_path, "Raw")):
        os.makedirs(os.path.join(config.result_path, "Raw"))
    if not os.path.exists(os.path.join(config.result_path, "Predict")):
        os.makedirs(os.path.join(config.result_path, "Predict"))
    if not os.path.exists(os.path.join(config.result_path, "GT")):
        os.makedirs(os.path.join(config.result_path, "GT"))
    if not os.path.exists(os.path.join(config.result_path, "Rescale")):
        os.makedirs(os.path.join(config.result_path, "Rescale"))
    
    solver = Solver(config)
    if config.mode == "train":
        train_dataset = LungCancerDataset(config.data_path,"train")
        valid_dataset = LungCancerDataset(config.data_path, "valid")

        train_dataloader = data.DataLoader(train_dataset, batch_size= config.batch_size, 
                                            shuffle=True,num_workers=config.num_workers)
        valid_dataloader = data.DataLoader(valid_dataset, batch_size= 1, 
                                            shuffle=False,num_workers=config.num_workers)
        solver.train(train_dataloader,valid_dataloader)
    else:
        test_dataset = LungCancerDataset(config.data_path,"test")
        test_dataloader = data.DataLoader(test_dataset, batch_size= 1, 
                                            shuffle=True,num_workers=config.num_workers)
        solver.test(test_dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # training hyper-parameterss
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.9)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    

    # misc
    parser.add_argument('--test_model', type=str, default='EV1')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--data_path', type=str, default='./Dataset')
    parser.add_argument('--result_path', type=str, default='./result/')

    parser.add_argument('--cuda_idx', type=int, default=0)

    config = parser.parse_args()
    print(config)
    main(config)