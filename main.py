import argparse
import os
from solver import Solver
from torch.backends import cudnn
from dicom_loader import get_dicom_loader
import random

def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net','NestedU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net/NestedU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return

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

    
    #lr = random.random()*0.0005 + 0.0000005
    augmentation_prob= random.random()*0.7
    #epoch = random.choice([100,150,200,250])
    # decay_ratio = random.random()*0.8
    # decay_epoch = int(epoch*decay_ratio)

    config.augmentation_prob = augmentation_prob
    #config.num_epochs = epoch
    # config.lr = lr
    # config.num_epochs_decay = decay_epoch

    #print(config)
    
    if config.mode == 'test':
        config.batch_size = 1 
        
    train_loader = get_dicom_loader(image_path=config.train_path,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                mode=config.mode,
                                augmentation_prob=config.augmentation_prob,
                                model=config.model_type)

    valid_loader = get_dicom_loader(image_path=config.valid_path,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                mode=config.mode,
                                augmentation_prob=config.augmentation_prob,
                                model=config.model_type)

    test_loader = get_dicom_loader(image_path=config.test_path,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                mode=config.mode,
                                augmentation_prob=config.augmentation_prob,
                                model=config.model_type)

    solver = Solver(config, train_loader, valid_loader, test_loader)

    
    # Train and sample the images
    if config.mode == 'train':
        print("*"*30,"Training Start","*"*30)
        solver.train()
    elif config.mode == 'test':
        print("*"*30,"Testing Start","*"*30)
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameterss
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--num_epochs_decay', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.9)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--test_model', type=str, default='EV1')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='NestedU_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--train_path', type=str, default='./dataset/train_dicom/')
    parser.add_argument('--valid_path', type=str, default='./dataset/valid_dicom/')
    parser.add_argument('--test_path', type=str, default='./dataset/eval/')
    parser.add_argument('--result_path', type=str, default='./result/')

    parser.add_argument('--cuda_idx', type=int, default=0)

    config = parser.parse_args()
    print(config)
    main(config)