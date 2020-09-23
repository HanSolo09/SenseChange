import torchvision.models as models
import torch
import torch.nn as nn
# Binary_DuoResUnet

def convrelu(in_channels, out_channels, kernel, padding):
  return nn.Sequential(nn.Conv2d(in_channels, out_channels,kernel,padding = padding),
                       nn.ReLU(inplace= True))
class DuoResNetUnet(nn.Module):
  def __init__(self, n_class):
    super().__init__()

    self.base_model = models.resnet18(pretrained = True)
    self.base_layers = list(self.base_model.children())
    # for child in self.base_layers:
    #   for param in child.parameters():
    #     param.requires_grad = False
    self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
    self.layer0_1x1 = convrelu(128, 64, 1, 0)
    self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
    self.layer1_1x1 = convrelu(128, 64, 1, 0)
    self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
    self.layer2_1x1 = convrelu(256, 128, 1, 0)
    self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
    self.layer3_1x1 = convrelu(512, 256, 1, 0)
    self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
    self.layer4_1x1 = convrelu(512, 512, 1, 0)
    self.layer_merge = convrelu(512+512,512,1,0)

    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
    self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
    self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
    self.conv_up0 = convrelu(64 + 256, 128, 3, 1)


    self.conv_original_size = convrelu(128, 64, 3, 1)

    self.conv_last = nn.Conv2d(64, n_class, 1)

  def forward(self, input1, input2):

    # input1 downconv
    x1_0 = self.layer0(input1) # -1,64,256,256
    x1_1 = self.layer1(x1_0)   # -1,64,128,128
    x1_2 = self.layer2(x1_1)   # -1,128,64,64
    x1_3 = self.layer3(x1_2)   # -1,256,32,32
    x1_4 = self.layer4(x1_3)   # -1,512,16,16

    # input2 downconv
    x2_0 = self.layer0(input2)
    x2_1 = self.layer1(x2_0)
    x2_2 = self.layer2(x2_1)
    x2_3 = self.layer3(x2_2)
    x2_4 = self.layer4(x2_3)
    
    # merge x1,x2
    x = torch.cat([x1_4, x2_4], dim=1) # -1,1024,16,16
    x = self.layer_merge(x) # -1,512,16,16
    
    #upconv
    # 32
    x = self.upsample(x) # -1,512,32,32
    x_3 = torch.cat([x1_3,x2_3],dim=1) # -1,512,32,32
    x_3 = self.layer3_1x1(x_3) # -1,256,32,32
    x = torch.cat([x,x_3],dim=1) # -1,256+512,32,32
    x = self.conv_up3(x) #-1,512,32,32
    
    # 64
    x = self.upsample(x) # -1,512, 64,64
    x_2 = torch.cat([x1_2,x2_2],dim=1) # -1,256,64,64
    x_2 = self.layer2_1x1(x_2) # -1, 128, 64, 64
    x = torch.cat([x,x_2],dim = 1) # -1, 128+512, 64,64
    x = self.conv_up2(x) #-1, 256, 64,64

    #128
    x = self.upsample(x) # -1, 256,128,128
    x_1 = torch.cat([x1_1,x2_1], dim = 1) # -1, 128, 128,128
    x_1 = self.layer1_1x1(x_1) # -1, 64, 128, 128
    x = torch.cat([x, x_1],dim = 1) # -1, 64+256, 128,128 
    x = self.conv_up1(x) # -1, 256, 128,128

    #256
    x = self.upsample(x) # -1, 256, 256,256
    x_0 = torch.cat([x1_0,x2_0],dim=1) #-1, 128, 256,256
    x_0 = self.layer0_1x1(x_0) # -1,64,256,256
    x = torch.cat([x,x_0],dim = 1) # -1,64+256,256,256
    x = self.conv_up0(x) # -1, 128,256,256

    #512
    x = self.upsample(x) #-1, 128,512,512
    x = self.conv_original_size(x) #-1,64,512,512
    x = self.conv_last(x)
    return x