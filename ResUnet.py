import torch
import torchvision.models as models
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np
import cv2
import glob
import torch
from torch.autograd import Variable
torch.cuda.empty_cache()

def convrelu(in_channels, out_channels, kernel, padding):
  return nn.Sequential(nn.Conv2d(in_channels, out_channels,kernel,padding = padding),
                       nn.ReLU(inplace= True))
class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out
class genDd(Dataset):
    def __init__(self,tifpatha,maskpatha,labelpath):
        self.tifDira = sorted(glob.glob(tifpatha))
        self.labelpath = sorted(glob.glob(labelpath))
        self.maskDir = sorted(glob.glob(maskpatha))
        
    def __getitem__(self,index):
        dta = cv2.imread(self.tifDira[index]).transpose(2,0,1)/255.0
        label = cv2.imread(self.labelpath[index],0)
        mk1 = cv2.imread(self.maskDir[index],0)
        
        for i in range(3):
          dta[i,:,:]=dta[i,:,:]*mk1

        return torch.from_numpy(dta).type(torch.FloatTensor),torch.from_numpy(label).type(torch.FloatTensor)
    def __len__(self):
        return len(self.tifDira)

# gsett = genDd("/shared_ssd/gaohan/SenseChange/train/im1/*.png",
#                "/shared_ssd/gaohan/SenseChange/train/label_mask/*.png",
#               "/shared_ssd/gaohan/SenseChange/train/label1/*.png")
# gloderr = DataLoader(gsett,batch_size=4,shuffle=True,num_workers=2)

# gsett2 = genDd("/shared_ssd/gaohan/SenseChange/train/im2/*.png",
#                "/shared_ssd/gaohan/SenseChange/train/label_mask/*.png",
#               "/shared_ssd/gaohan/SenseChange/train/label2/*.png")

# gloderr2 = DataLoader(gsett2,batch_size=4,shuffle=True,num_workers=2)

# device = torch.device("cuda:3")
# net = ResNetUNet(7).to(device)
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=1e-4,weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

##loss
def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

class CrossEntropyLoss2d(nn.Module):
  def __init__(self, weight=None, ignore_index=255, reduction='mean'):
      super(CrossEntropyLoss2d, self).__init__()
      self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

  def forward(self, output, target):
      loss = self.CE(output, target)
      return loss
class DiceLoss(nn.Module):

  def __init__(self, smooth=1., ignore_index=255):
      super(DiceLoss, self).__init__()
      self.ignore_index = ignore_index
      self.smooth = smooth

  def forward(self, output, target):
      if self.ignore_index not in range(target.min(), target.max()):
          if (target == self.ignore_index).sum() > 0:
              target[target == self.ignore_index] = target.min()
      target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
      output = F.softmax(output, dim=1)
      output_flat = output.contiguous().view(-1)
      target_flat = target.contiguous().view(-1)
      intersection = (output_flat * target_flat).sum()
      loss = 1 - ((2. * intersection + self.smooth) /
                  (output_flat.sum() + target_flat.sum() + self.smooth))
      return loss
class CE_DiceLoss(nn.Module):

  def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
      super(CE_DiceLoss, self).__init__()
      self.smooth = smooth
      self.dice = DiceLoss()
      self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
  
  def forward(self, output, target):
      CE_loss = self.cross_entropy(output, target)
      dice_loss = self.dice(output, target)
      return CE_loss + dice_loss

# CEloss = CE_DiceLoss()



def train():
	print('start training')
	for e in range(100):
		epoch_loss = []
		for dd1,dd2 in gloderr:
		    optimizer.zero_grad()
		    imgs1 = Variable(dd1).to(device,dtype=torch.float32)
		    label =Variable(dd2).to(device,dtype=torch.long)
		    # mks1 = Variable(mm1).to(device,dtype=torch.long)
		    pred1 = net(imgs1)
		    loss = CEloss(pred1,label)
		    loss.backward()
		    optimizer.step()
		    epoch_loss.append(loss.item())

		for dd1,dd2 in gloderr2:
		    optimizer.zero_grad()
		    imgs1 = Variable(dd1).to(device,dtype=torch.float32)
		    label =Variable(dd2).to(device,dtype=torch.long)
		    # mks1 = Variable(mm1).to(device,dtype=torch.long)
		    pred1 = net(imgs1)
		    loss = CEloss(pred1,label )
		    loss.backward()
		    optimizer.step()
		    epoch_loss.append(loss.item())
	    
	    
		scheduler.step()
		# print(np.mean(np.array(epoch_loss),0))
		print(np.mean(np.array(epoch_loss),0))
		torch.save(net.state_dict(),"/shared_ssd/gaohan/SenseChange/weight/ce_mask_class.pth")

device = torch.device("cuda:2")
net = ResNetUNet(7).to(device)
net.load_state_dict(torch.load("/shared_ssd/gaohan/SenseChange/weight/ce_mask_class.pth"))
def mgtSia1(net1,img1,msk):
	img = cv2.imread(img1).transpose(2,0,1)/255.0
	mk = cv2.imread(msk,0)
	for i in range(3):
		img[i,:,:]=img[i,:,:]*mk
	img = np.expand_dims(img,0)
	predData = Variable(torch.from_numpy(np.array(img,dtype=np.float32))).to(device)
	pred1= net1(predData)
	im1 = torch.squeeze(pred1).argmax(0).data.cpu().numpy()
	return im1

def genvaldata(val1_dir, val2_dir,mask_dir):



	dir1_list= sorted(glob.glob(val1_dir))
	dir2_list = sorted(glob.glob(val2_dir))
	mk_list = sorted(glob.glob(mask_dir))

	num = len(dir1_list)
	print(dir2_list[0])
	i = 0
	for i in range(len(dir1_list)):


		img1_name = dir1_list[i].split('/')[-1]
		img2_name = dir2_list[i].split('/')[-1]
		if img1_name == img2_name:
		  # pred1,pred2 = mgtSia1(net2,dir1_list[i],dir2_list[i])
		  pred1 = mgtSia1(net,dir1_list[i],mk_list[i])
		  pred2 = mgtSia1(net, dir2_list[i],mk_list[i])


		  path1 = '/shared_ssd/gaohan/SenseChange/pred/pred_921/im1/'+img1_name
		  path2 = '/shared_ssd/gaohan/SenseChange/pred/pred_921/im2/'+img2_name
		  cv2.imwrite(path1,pred1)
		  cv2.imwrite(path2,pred2)
		  print((num-i))
		else:
		  print('error',img1_name,img2_name)

val1_dir = '/shared_ssd/gaohan/SenseChange/val/im1/*.png'
val2_dir = '/shared_ssd/gaohan/SenseChange/val/im2/*.png'
mask_dir = '/shared_ssd/gaohan/SenseChange/pred/mask_pred/merge/*.png'

print('start pred')

genvaldata(val1_dir, val2_dir,mask_dir)

