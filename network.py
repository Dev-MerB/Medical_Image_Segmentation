import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import scipy.io as sio
from torch.nn.modules import padding


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class U_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2U_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



class AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,t=2):
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class conv_block_nested(nn.Module):
    
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output
    
#Nested Unet

class Nested_UNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, img_ch=1, output_ch=1):
        super(Nested_UNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(img_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], output_ch, kernel_size=1)


    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output

# class MIA(nn.Module):
#     def __init__(self,img_ch=1,output_ch=1):
#         super(MIA, self).__init__()
#         self.conv1 = torch.nn.Sequential(
#             torch.nn.Conv2d(img_ch,64,kernel_size=3,stride=1,padding=35)
#         )

#         self.deconv_1 = nn.ConvTranspose2d(1, 1, (8,8), padding=4, stride=4)
#         self.deconv_2 = nn.ConvTranspose2d(1, 1, (16,16), padding=8, stride=8)
#         self.deconv_3 = nn.ConvTranspose2d(1, 1, (32,32), padding=16, stride=16)

#         # self.score_dsn1 = nn.Conv2d(64, 1, 1)
#         # self.score_dsn2 = nn.Conv2d(128, 1, 1)
#         self.score_dsn3 = nn.Conv2d(256, 1, 1)
#         self.score_dsn4 = nn.Conv2d(512, 1, 1)
#         self.score_dsn5 = nn.Conv2d(512, 1, 1)
#         self.score_final = nn.Conv2d(1, 1, 1)

#     def forward(self, x):
#         # VGG
#         img_H, img_W = x.shape[2], x.shape[3]
#         conv1_1 = self.relu(self.conv1_1(x))
#         conv1_2 = self.relu(self.conv1_2(conv1_1))
#         pool1 = self.maxpool(conv1_2)

#         conv2_1 = self.relu(self.conv2_1(pool1))
#         conv2_2 = self.relu(self.conv2_2(conv2_1))
#         pool2 = self.maxpool(conv2_2)

#         conv3_1 = self.relu(self.conv3_1(pool2))
#         conv3_2 = self.relu(self.conv3_2(conv3_1))
#         conv3_3 = self.relu(self.conv3_3(conv3_2))
#         pool3 = self.maxpool(conv3_3)

#         conv4_1 = self.relu(self.conv4_1(pool3))
#         conv4_2 = self.relu(self.conv4_2(conv4_1))
#         conv4_3 = self.relu(self.conv4_3(conv4_2))
#         pool4 = self.maxpool(conv4_3)

#         conv5_1 = self.relu(self.conv5_1(pool4))
#         conv5_2 = self.relu(self.conv5_2(conv5_1))
#         conv5_3 = self.relu(self.conv5_3(conv5_2))

#         # so1 = self.score_dsn1(conv1_2)
#         # so2 = self.score_dsn2(conv2_2)
#         so3 = self.score_dsn3(conv3_3)
#         so4 = self.score_dsn4(conv4_3)
#         so5 = self.score_dsn5(conv5_3)
#         # upsample1 = self.deconv_1(so3)
#         # upsample2 = self.deconv_2(so4)
#         # upsample3 = self.deconv_3(so5)
#         # print(so5.shape, so4.shape, so3.shape)
#         # print(upsample3.shape, upsample2.shape, upsample1.shape)

#         # #weight_deconv2 =  make_bilinear_weights(4, 1).cuda()
#         weight_deconv3 =  make_bilinear_weights(8, 1).cuda()
#         weight_deconv4 =  make_bilinear_weights(16, 1).cuda()
#         weight_deconv5 =  make_bilinear_weights(32, 1).cuda()

#         # upsample2 = torch.nn.functional.conv_transpose2d(so2, weight_deconv2, stride=2)
#         upsample3 = torch.nn.functional.conv_transpose2d(so3, weight_deconv3, stride=4)
#         upsample4 = torch.nn.functional.conv_transpose2d(so4, weight_deconv4, stride=8)
#         upsample5 = torch.nn.functional.conv_transpose2d(so5, weight_deconv5, stride=16)

#         # so1 = crop(so1, img_H, img_W)
#         # so2 = crop(upsample2, img_H, img_W)
#         so3 = crop(upsample3, img_H, img_W)
#         so4 = crop(upsample4, img_H, img_W)
#         so5 = crop(upsample5, img_H, img_W)
#         # fusecat = torch.mul((so3, so4, so5), dim=1)
#         # fusecat = torch.cat((so3, so4, so5), dim=1)
#         fusecat = torch.sum(torch.cat((so3, so4, so5), dim=1), dim=1)
#         fusecat = fusecat.unsqueeze(1)
#         # print(fusecat.shape)
#         fuse = self.score_final(fusecat)

#         # results = [so3, so4, so5, fuse]
#         # results = [torch.sigmoid(r) for r in results]
#         return fuse

# class MIA(nn.Module):
#     def __init__(self,img_ch=1,output_ch=1):
#         super(MIA, self).__init__()
#         self.conv1_1 = nn.Conv2d(img_ch, 64, 3, padding=35)
#         self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

#         self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
#         self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

#         self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
#         self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
#         self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

#         self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
#         self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
#         self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)

#         self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
#         self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
#         self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)

#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

#         self.deconv_1 = nn.ConvTranspose2d(1, 1, (8,8), padding=4, stride=4)
#         self.deconv_2 = nn.ConvTranspose2d(1, 1, (16,16), padding=8, stride=8)
#         self.deconv_3 = nn.ConvTranspose2d(1, 1, (32,32), padding=16, stride=16)

#         # self.score_dsn1 = nn.Conv2d(64, 1, 1)
#         # self.score_dsn2 = nn.Conv2d(128, 1, 1)
#         self.score_dsn3 = nn.Conv2d(256, 1, 1)
#         self.score_dsn4 = nn.Conv2d(512, 1, 1)
#         self.score_dsn5 = nn.Conv2d(512, 1, 1)
#         self.score_final = nn.Conv2d(1, 1, 1)

#     def forward(self, x):
#         # VGG
#         img_H, img_W = x.shape[2], x.shape[3]
#         conv1_1 = self.relu(self.conv1_1(x))
#         conv1_2 = self.relu(self.conv1_2(conv1_1))
#         pool1 = self.maxpool(conv1_2)

#         conv2_1 = self.relu(self.conv2_1(pool1))
#         conv2_2 = self.relu(self.conv2_2(conv2_1))
#         pool2 = self.maxpool(conv2_2)

#         conv3_1 = self.relu(self.conv3_1(pool2))
#         conv3_2 = self.relu(self.conv3_2(conv3_1))
#         conv3_3 = self.relu(self.conv3_3(conv3_2))
#         pool3 = self.maxpool(conv3_3)

#         conv4_1 = self.relu(self.conv4_1(pool3))
#         conv4_2 = self.relu(self.conv4_2(conv4_1))
#         conv4_3 = self.relu(self.conv4_3(conv4_2))
#         pool4 = self.maxpool(conv4_3)

#         conv5_1 = self.relu(self.conv5_1(pool4))
#         conv5_2 = self.relu(self.conv5_2(conv5_1))
#         conv5_3 = self.relu(self.conv5_3(conv5_2))

#         # so1 = self.score_dsn1(conv1_2)
#         # so2 = self.score_dsn2(conv2_2)
#         so3 = self.score_dsn3(conv3_3)
#         so4 = self.score_dsn4(conv4_3)
#         so5 = self.score_dsn5(conv5_3)
#         # upsample1 = self.deconv_1(so3)
#         # upsample2 = self.deconv_2(so4)
#         # upsample3 = self.deconv_3(so5)
#         # print(so5.shape, so4.shape, so3.shape)
#         # print(upsample3.shape, upsample2.shape, upsample1.shape)

#         # #weight_deconv2 =  make_bilinear_weights(4, 1).cuda()
#         weight_deconv3 =  make_bilinear_weights(8, 1).cuda()
#         weight_deconv4 =  make_bilinear_weights(16, 1).cuda()
#         weight_deconv5 =  make_bilinear_weights(32, 1).cuda()

#         # upsample2 = torch.nn.functional.conv_transpose2d(so2, weight_deconv2, stride=2)
#         upsample3 = torch.nn.functional.conv_transpose2d(so3, weight_deconv3, stride=4)
#         upsample4 = torch.nn.functional.conv_transpose2d(so4, weight_deconv4, stride=8)
#         upsample5 = torch.nn.functional.conv_transpose2d(so5, weight_deconv5, stride=16)

#         # so1 = crop(so1, img_H, img_W)
#         # so2 = crop(upsample2, img_H, img_W)
#         so3 = crop(upsample3, img_H, img_W)
#         so4 = crop(upsample4, img_H, img_W)
#         so5 = crop(upsample5, img_H, img_W)
#         # fusecat = torch.mul((so3, so4, so5), dim=1)
#         # fusecat = torch.cat((so3, so4, so5), dim=1)
#         fusecat = torch.sum(torch.cat((so3, so4, so5), dim=1), dim=1)
#         fusecat = fusecat.unsqueeze(1)
#         # print(fusecat.shape)
#         fuse = self.score_final(fusecat)

#         # results = [so3, so4, so5, fuse]
#         # results = [torch.sigmoid(r) for r in results]
#         return fuse


# def crop(variable, th, tw):
#     h, w = variable.shape[2], variable.shape[3]
#     x1 = int(round((w - tw) / 2.))
#     y1 = int(round((h - th) / 2.))
#     return variable[:, :, y1: y1 + th, x1: x1 + tw]


# # make a bilinear interpolation kernel
# def upsample_filt(size):
#     factor = (size + 1) // 2
#     if size % 2 == 1:
#         center = factor - 1
#     else:
#         center = factor - 0.5
#     og = np.ogrid[:size, :size]
#     return (1 - abs(og[0] - center) / factor) * \
#            (1 - abs(og[1] - center) / factor)


# # set parameters s.t. deconvolutional layers compute bilinear interpolation
# # N.B. this is for deconvolution without groups
# def interp_surgery(in_channels, out_channels, h, w):
#     weights = np.zeros([in_channels, out_channels, h, w])
#     if in_channels != out_channels:
#         raise ValueError("Input Output channel!")
#     if h != w:
#         raise ValueError("filters need to be square!")
#     filt = upsample_filt(h)
#     weights[range(in_channels), range(out_channels), :, :] = filt
#     return np.float32(weights)


# def make_bilinear_weights(size, num_channels):
#     factor = (size + 1) // 2
#     if size % 2 == 1:
#         center = factor - 1
#     else:
#         center = factor - 0.5
#     og = np.ogrid[:size, :size]
#     filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
#     # print(filt)
#     filt = torch.from_numpy(filt)
#     w = torch.zeros(num_channels, num_channels, size, size)
#     w.requires_grad = False
#     for i in range(num_channels):
#         for j in range(num_channels):
#             if i == j:
#                 w[i, j] = filt
#     return w


# def upsample(input, stride, num_channels=1):
#     kernel_size = stride * 2
#     kernel = make_bilinear_weights(kernel_size, num_channels).cuda()
#     return torch.nn.functional.conv_transpose2d(input, kernel, stride=stride)