"""
XD_XD's solution for the SpaceNet Off-Nadir Building Detection Challenge
** Usage
 $ python main.py <sub-command> [options]
    sub-commands:
        - train
        - inference
        - filecheck
        - check
"""
import warnings
from pathlib import Path

import csv
import os
import datetime
import json
import shutil
import time
import sys

# import scipy.sparse as ss
import numpy as np
import pandas as pd
import attr
import click
import tqdm
import cv2

import skimage.measure
from sklearn.utils import Bunch

from torch import nn
import torch
from torch.optim import Adam
from torchvision.models import vgg16
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler
# from albumentations.torch.functional import img_to_tensor
from albumentations.pytorch.transforms import img_to_tensor
from albumentations import (
    Normalize, Compose, HorizontalFlip, RandomRotate90, RandomCrop, VerticalFlip,
    Transpose,RandomBrightnessContrast
    )
import torchvision.models as models
import scipy.sparse as ss
import tempfile
# from models import DuoResNetUnet

# import spacenetutilities.labeltools.coreLabelTools as cLT
# from spacenetutilities import geoTools as gT
# from shapely.geometry import shape
# from shapely.wkt import dumps
# import geopandas as gpd


warnings.simplefilter(action='ignore', category=FutureWarning)

# creat models
# class conv_relu(nn.Module):
#     def __init__(self, in_, out):
#         super().__init__()
#         self.conv = nn.Conv2d(in_, out, 3, padding=1)
#         self.activation = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.activation(x)
#         return x


# class decoder_block(nn.Module):
#     def __init__(self, in_channels, middle_channels, out_channels):
#         super(decoder_block, self).__init__()
#         self.in_channels = in_channels
#         self.block = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             conv_relu(in_channels, middle_channels),
#             conv_relu(middle_channels, out_channels),
#         )

#     def forward(self, x):
#         return self.block(x)


# class unet_vgg16(nn.Module):
#     def __init__(self, num_filters=32, pretrained=False):
#         super().__init__()
#         self.encoder = vgg16(pretrained=pretrained).features
#         self.pool = nn.MaxPool2d(2, 2)

#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.Sequential(
#             self.encoder[0], self.relu, self.encoder[2], self.relu)
#         self.conv2 = nn.Sequential(
#             self.encoder[5], self.relu, self.encoder[7], self.relu)
#         self.conv3 = nn.Sequential(
#             self.encoder[10], self.relu, self.encoder[12], self.relu,
#             self.encoder[14], self.relu)
#         self.conv4 = nn.Sequential(
#             self.encoder[17], self.relu, self.encoder[19], self.relu,
#             self.encoder[21], self.relu)
#         self.conv5 = nn.Sequential(
#             self.encoder[24], self.relu, self.encoder[26], self.relu,
#             self.encoder[28], self.relu)

#         self.center = decoder_block(512, num_filters * 8 * 2, num_filters * 8)
#         self.dec5 = decoder_block(
#             512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
#         self.dec4 = decoder_block(
#             512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
#         self.dec3 = decoder_block(
#             256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
#         self.dec2 = decoder_block(
#             128 + num_filters * 2, num_filters * 2 * 2, num_filters)
#         self.dec1 = conv_relu(64 + num_filters, num_filters)
#         self.final = nn.Conv2d(num_filters, 1, kernel_size=1)

#     def forward(self, x):
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(self.pool(conv1))
#         conv3 = self.conv3(self.pool(conv2))
#         conv4 = self.conv4(self.pool(conv3))
#         conv5 = self.conv5(self.pool(conv4))
#         center = self.center(self.pool(conv5))
#         dec5 = self.dec5(torch.cat([center, conv5], 1))
#         dec4 = self.dec4(torch.cat([dec5, conv4], 1))
#         dec3 = self.dec3(torch.cat([dec4, conv3], 1))
#         dec2 = self.dec2(torch.cat([dec3, conv2], 1))
#         dec1 = self.dec1(torch.cat([dec2, conv1], 1))
#         x_out = self.final(dec1)
#         return x_out
def convrelu(in_channels, out_channels, kernel, padding):
  return nn.Sequential(nn.Conv2d(in_channels, out_channels,kernel,padding = padding),
                       nn.ReLU(inplace= True))
class DuoResNetUnet(nn.Module):
  def __init__(self, n_class):
    super().__init__()

    self.base_model = models.resnet18(pretrained = False)
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
DATABASE = '/shared_ssd/gaohan/SenseChange/train'
HOME = '/shared_ssd/gaohan/SenseChange/train'
# build dataloader
def get_image(imageid, 
    basepath= DATABASE,
    time ='im1'):
    fn = f'{basepath}/{time}/{imageid}.png'
    
    img = cv2.imread(fn, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
    return img

# TODO: 修改读入图像路径
class ChangeDataset(Dataset):
    def __init__(self, image_ids, aug=None, basepath=DATABASE):
        self.image_ids = image_ids
        self.aug = aug
        self.basepath = basepath

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        imageid = self.image_ids[idx]
        im1= get_image(imageid, basepath=self.basepath, time ='im1')
        im2 = get_image(imageid, basepath = self.basepath, time = 'im2')

        mask = cv2.imread(
            f'{self.basepath}/mask/{imageid}.png',
            cv2.IMREAD_GRAYSCALE)
        assert mask is not None
        
        augmented = self.aug(image = im1, image1 = im2, mask = mask)

        mask_ = (augmented['mask'] > 0).astype(np.uint8)
        mask_ = torch.from_numpy(np.expand_dims(mask_, 0)).float()
        # label_ = torch.from_numpy(np.expand_dims(augmented['mask'], 0)).float()

        return (
            img_to_tensor(augmented['image']), 
            img_to_tensor(augmented['image1']),
            mask_,  
            imageid)


class ChangeTestDataset(Dataset):
    def __init__(self, image_ids, aug=None, basepath='/shared_ssd/gaohan/SenseChange/train'):
        self.image_ids = image_ids
        self.aug = aug
        self.basepath = basepath

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        imageid = self.image_ids[idx]
        im1= get_image(imageid, basepath=self.basepath, time ='im1')
        im2 = get_image(imageid, basepath = self.basepath, time = 'im2')
        

        augmented = self.aug(image=im1,image1 = im2)
        return img_to_tensor(augmented['image']), img_to_tensor(augmented['image1']),imageid


@click.group()
def cli():
    pass


@cli.command()
@click.option('--inputs', '-i', default='/data/test',
              help='input directory')
def check(inputs):
    systemcheck_train()
    # TODO: check training images


# @cli.command()
# @click.option('--inputs', '-i', default='/data/test',
#               help='input directory')
# @click.option('--working_dir', '-w', default='/wdata',
#               help="working directory")
# def preproctrain(inputs, working_dir):
#     """
#     * Making 8bit rgb train images
#     """
#     # preproc images
#     Path(f'{working_dir}/dataset/train_rgb').mkdir(parents=True,
#                                                    exist_ok=True)
#     catalog_paths = list(sorted(Path(inputs).glob('./Atlanta_nadir*')))
#     assert len(catalog_paths) > 0
#     print('Found {} catalog directories'.format(len(catalog_paths)))
#     for catalog_dir in tqdm.tqdm(catalog_paths, total=len(catalog_paths)):
#         src_imgs = list(sorted(catalog_dir.glob('./Pan-Sharpen/Pan-*.tif')))
#         for src in tqdm.tqdm(src_imgs, total=len(src_imgs)):
#             dst = f'{working_dir}/dataset/train_rgb/{src.name}'
#             if not Path(dst).exists():
#                 pan_to_bgr(str(src), dst)

#     # prerpoc masks
#     (Path(working_dir) / Path('dataset/masks')).mkdir(parents=True,
#                                                       exist_ok=True)
#     geojson_dir = Path(inputs) / Path('geojson/spacenet-buildings')
#     mask_dir = Path(working_dir) / Path('dataset/masks')
#     ref_catalog_name = list(Path(inputs).glob(
#         './Atlanta_nadir*/Pan-Sharpen'))[0].parent.name
#     for geojson_fn in geojson_dir.glob('./spacenet-buildings_*.geojson'):
#         masks_from_geojson(mask_dir, inputs, ref_catalog_name, geojson_fn)


# def masks_from_geojson(mask_dir, inputs, ref_name, geojson_fn):
#     chip_id = geojson_fn.name.lstrip('spacenet-buildings_').rstrip('.geojson')
#     mask_fn = mask_dir / f'mask_{chip_id}.tif'
#     if mask_fn.exists():
#         return

#     ref_fn = str(Path(inputs) / Path(
#         f'{ref_name}/Pan-Sharpen/Pan-Sharpen_{ref_name}_{chip_id}.tif'))
#     cLT.createRasterFromGeoJson(str(geojson_fn), ref_fn, str(mask_fn))

# 返回图像编号、路径、时间和fold id

def read_cv_splits(inputs):
    fn = '/shared_ssd/gaohan/SenseChange/train/'+'cv2.txt'
    if not Path(fn).exists():
        train_imageids = list(sorted(
            Path(inputs).glob('./im1/*.png')))

        # split 4 folds
        df_fold = pd.DataFrame({
            'filename': train_imageids,
            'time': [path.parent.name for path in train_imageids],
        })
        df_fold.loc[:, 'fold_id'] = np.random.randint(0, 5, len(df_fold))
        df_fold.loc[:, 'ImageId'] = df_fold.filename.apply(
            lambda x: x.name[0:-4])

        df_fold[[
            'ImageId', 'filename', 'time', 'fold_id',
        ]].to_csv(fn, index=False)

    return pd.read_csv(fn,converters={'ImageId': str})


@cli.command()  # noqa: C901
@click.option('--inputs', '-i', default='/shared_ssd/gaohan/SenseChange/train',
              help='input directory')
@click.option('--working_dir', '-w', default='./working',
              help="working directory")
@click.option('--fold_id', '-f', default=0, help='fold id')
def train(inputs, working_dir, fold_id):
    start_epoch, step = 0, 0

    # TopCoder
    num_workers, batch_size = 8, 4 * 4
    gpus = [0, 1, 2, 3]

    # My machine
    # num_workers, batch_size = 8, 2 * 3
    # gpus = [0, 1]

    patience, n_epochs = 8, 150
    lr, min_lr, lr_update_rate = 1e-4, 5e-5, 0.5
    training_timelimit = 60 * 60 * 24 * 2  # 2 days
    st_time = time.time()

    # model = unet_vgg16(pretrained=True)
    model = DuoResNetUnet(1)
    print('model load success')
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    train_transformer = Compose([
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
        VerticalFlip(p = 0.2),
        Normalize(),
    ], p=1.0,additional_targets = {'image1' :'image'})

    val_transformer = Compose([
        Normalize(),
    ], p=1.0,additional_targets = {'image1' :'image'})

    # train/val loadrs
    df_cvfolds = read_cv_splits(inputs)
   
    trn_loader, val_loader = make_train_val_loader(
        train_transformer, val_transformer, df_cvfolds, fold_id,
        batch_size, num_workers)

    # train
    criterion = binary_loss(jaccard_weight=0.25)
    optimizer = Adam(model.parameters(), lr=lr)

    report_epoch = 10

    model_name = f'BiResUnet_f{fold_id}'
    fh = open_log(model_name)

    # vers for early stopping
    best_score = 0
    not_improved_count = 0

    for epoch in range(start_epoch, n_epochs):
        model.train()

        tl = trn_loader  # alias
        trn_metrics = Metrics()

        try:
            tq = tqdm.tqdm(total=(len(tl) * trn_loader.batch_size))
            tq.set_description(f'Ep{epoch:>3d}')
            for i, (im1, im2, mask, names) in enumerate(trn_loader):
                im1 = im1.cuda()
                im2 = im2.cuda()
                mask = mask.cuda()

                outputs = model(im1,im2)
                loss = criterion(outputs, mask)
                optimizer.zero_grad()

                # Increment step counter
                batch_size = im1.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)

                # Update eval metrics
                trn_metrics.loss.append(loss.item())
                trn_metrics.bce.append(criterion._stash_bce_loss.item())
                trn_metrics.jaccard.append(criterion._stash_jaccard.item())

                if i > 0 and i % report_epoch == 0:
                    report_metrics = Bunch(
                        epoch=epoch,
                        step=step,
                        trn_loss=np.mean(trn_metrics.loss[-report_epoch:]),
                        trn_bce=np.mean(trn_metrics.bce[-report_epoch:]),
                        trn_jaccard=np.mean(
                            trn_metrics.jaccard[-report_epoch:]),
                    )
                    write_event(fh, **report_metrics)
                    tq.set_postfix(
                        loss=f'{report_metrics.trn_loss:.5f}',
                        bce=f'{report_metrics.trn_bce:.5f}',
                        jaccard=f'{report_metrics.trn_jaccard:.5f}')

            # End of epoch
            report_metrics = Bunch(
                epoch=epoch,
                step=step,
                trn_loss=np.mean(trn_metrics.loss[-report_epoch:]),
                trn_bce=np.mean(trn_metrics.bce[-report_epoch:]),
                trn_jaccard=np.mean(trn_metrics.jaccard[-report_epoch:]),
            )
            write_event(fh, **report_metrics)
            tq.set_postfix(
                loss=f'{report_metrics.trn_loss:.5f}',
                bce=f'{report_metrics.trn_bce:.5f}',
                jaccard=f'{report_metrics.trn_jaccard:.5f}')
            tq.close()
            save(model, epoch, step, model_name)

            # Run validation
            val_metrics = validation(model,
                                     criterion,
                                     val_loader,
                                     epoch,
                                     step,
                                     fh)
            report_val_metrics = Bunch(
                epoch=epoch,
                step=step,
                val_loss=np.mean(val_metrics.loss[-report_epoch:]),
                val_bce=np.mean(val_metrics.bce[-report_epoch:]),
                val_jaccard=np.mean(val_metrics.jaccard[-report_epoch:]),
            )
            write_event(fh, **report_val_metrics)

            if time.time() - st_time > training_timelimit:
                tq.close()
                break

            if best_score < report_val_metrics.val_jaccard:
                best_score = report_val_metrics.val_jaccard
                not_improved_count = 0
                copy_best(model, epoch, model_name, step)
            else:
                not_improved_count += 1

            if not_improved_count >= patience:
                # Update learning rate and optimizer

                lr *= lr_update_rate
                # Stop criterion
                if lr < min_lr:
                    tq.close()
                    print('get min_lr')
                    break

                not_improved_count = 0

                # Load best weight
                del model
                model = DuoResNetUnet(1)
                path = f'/shared_ssd/gaohan/SenseChange/train/weights/{model_name}/{model_name}_best'
                cp = torch.load(path)
                model = nn.DataParallel(model).cuda()
                epoch = cp['epoch']
                model.load_state_dict(cp['model'])
                model = model.module
                model = nn.DataParallel(model, device_ids=gpus).cuda()

                # Init optimizer
                optimizer = Adam(model.parameters(), lr=lr)

        except KeyboardInterrupt:
            save(model, epoch, step, model_name)
            tq.close()
            fh.close()
            sys.exit(1)
        except Exception as e:
            raise e
            break

    fh.close()


def validation(model, criterion, val_loader,
               epoch, step, fh):
    report_epoch = 10
    val_metrics = Metrics()

    with torch.no_grad():
        model.eval()

        vl = val_loader

        tq = tqdm.tqdm(total=(len(vl) * val_loader.batch_size))
        tq.set_description(f'(val) Ep{epoch:>3d}')
        for i, (im1, im2, mask, names) in enumerate(val_loader):
            im1 = im1.cuda()
            im2 = im2.cuda()
            mask = mask.cuda()

            outputs = model(im1,im2)
            loss = criterion(outputs, mask)
            tq.update(im1.size(0))

            val_metrics.loss.append(loss.item())
            val_metrics.bce.append(criterion._stash_bce_loss.item())
            val_metrics.jaccard.append(criterion._stash_jaccard.item())

            if i > 0 and i % report_epoch == 0:
                report_metrics = Bunch(
                    epoch=epoch,
                    step=step,
                    val_loss=np.mean(val_metrics.loss[-report_epoch:]),
                    val_bce=np.mean(val_metrics.bce[-report_epoch:]),
                    val_jaccard=np.mean(
                        val_metrics.jaccard[-report_epoch:]),
                )
                tq.set_postfix(
                    loss=f'{report_metrics.val_loss:.5f}',
                    bce=f'{report_metrics.val_bce:.5f}',
                    jaccard=f'{report_metrics.val_jaccard:.5f}')

        # End of epoch
        report_metrics = Bunch(
            epoch=epoch,
            step=step,
            val_loss=np.mean(val_metrics.loss[-report_epoch:]),
            val_bce=np.mean(val_metrics.bce[-report_epoch:]),
            val_jaccard=np.mean(val_metrics.jaccard[-report_epoch:]),
        )
        tq.set_postfix(
            loss=f'{report_metrics.val_loss:.5f}',
            bce=f'{report_metrics.val_bce:.5f}',
            jaccard=f'{report_metrics.val_jaccard:.5f}')
        tq.close()

    return val_metrics


@attr.s
class Metrics(object):
    loss = attr.ib(default=[])
    bce = attr.ib(default=[])
    jaccard = attr.ib(default=[])


class binary_loss(object):
    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight
        self._stash_bce_loss = 0
        self._stash_jaccard = 0

    def __call__(self, outputs, targets):
        eps = 1e-15

        self._stash_bce_loss = self.nll_loss(outputs, targets)
        loss = (1 - self.jaccard_weight) * self._stash_bce_loss

        jaccard_target = (targets == 1).float()
        jaccard_output = torch.sigmoid(outputs)

        intersection = (jaccard_output * jaccard_target).sum()
        union = jaccard_output.sum() + jaccard_target.sum()

        jaccard_score = (
            (intersection + eps) / (union - intersection + eps))
        self._stash_jaccard = jaccard_score
        loss += self.jaccard_weight * (1. - jaccard_score)

        return loss


def save(model, epoch, step, model_name):
    path = f'/shared_ssd/gaohan/SenseChange/train/weights/{model_name}/{model_name}_ep{epoch}_{step}'
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'step': step,
    }, path)


def copy_best(model, epoch, model_name, step):
    path = f'/shared_ssd/gaohan/SenseChange/train/weights/{model_name}/{model_name}_ep{epoch}_{step}'
    best_path = f'/shared_ssd/gaohan/SenseChange/train/weights/{model_name}/{model_name}_best'
    shutil.copy(path, best_path)


def write_event(log, **data):
    data['dt'] = datetime.datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def open_log(model_name):
    time_str = datetime.datetime.now().strftime('%Y%m%d.%H%M%S')
    path = f'/shared_ssd/gaohan/SenseChange/train/weights/{model_name}/{model_name}.{time_str}.log'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, 'at', encoding='utf8')
    return fh


def make_train_val_loader(train_transformer,
                          val_transformer,
                          df_cvfolds,
                          fold_id,
                          batch_size,
                          num_workers):
    trn_dataset = ChangeDataset(
        df_cvfolds[df_cvfolds.fold_id != fold_id].ImageId.tolist(),
        aug=train_transformer)
    trn_loader = DataLoader(
        trn_dataset,
        sampler=RandomSampler(trn_dataset),
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available())

    val_dataset = ChangeDataset(
        df_cvfolds[df_cvfolds.fold_id == fold_id].ImageId.tolist(),
        aug=val_transformer)
    val_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available())
    return trn_loader, val_loader


@cli.command()
@click.option('--inputs', '-i', default='/shared_ssd/gaohan/SenseChange/val',
              help='input directory')
def inference(inputs):
    print('Collect filenames...')
    test_collection = []
    test_collection = list(sorted(Path(inputs).glob('./im1/*.png')))
    assert len(test_collection) > 0
    model_names = [
        'BiResUnet_f0_best',
        'BiResUnet_f1_best',
        'BiResUnet_f2_best',
        'BiResUnet_f3_best',
        'BiResUnet_f4_best'
    ]
    image_ids = list(map(lambda x: x.name[0:-4],test_collection))
    for model_name in model_names:
        inference_by_model(model_name, image_ids)

    # merge prediction masks and write submission file
    
    make_sub(model_names, image_ids)

def inference_by_model(model_name, ids,
                       batch_size=1,
                       num_workers=0,
                       fullsize_mode=False):
    # TODO: Optimize parameters for p2.xlarge
    print(f'Inrefernce by {model_name}')
    prefix = '_'.join(model_name.split('_')[:2]) # return  BiResUnet_f0
    model_checkpoint_file = f'/shared_ssd/gaohan/SenseChange/train/weights/{prefix}/{model_name}'

    pred_mask_dir = f'/shared_ssd/gaohan/SenseChange/pred/mask_pred/test_{model_name}/'
    Path(pred_mask_dir).mkdir(parents=True, exist_ok=True)

    model = DuoResNetUnet(1)
    print(model_checkpoint_file)
    cp = torch.load(model_checkpoint_file)
    model = nn.DataParallel(model).cuda()
    epoch = cp['epoch']
    model.load_state_dict(cp['model'])
    model = model.module
    model = model.cuda()

    print('model load success')

    # image_ids = [
    #     Path(path).name.lstrip('Pan-Sharpen_').rstrip('.tif')
    #     for path in Path('/wdata/dataset/test_rgb/').glob(
    #         'Pan-Sharpen*.tif')]
    image_ids = ids

    tst_transformer = Compose([
        Normalize(),
    ], p=1.0,additional_targets = {'image1' :'image'})
    tst_dataset = ChangeTestDataset(image_ids, aug=tst_transformer)
    tst_loader = DataLoader(
        tst_dataset,
        sampler=SequentialSampler(tst_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available())

    with torch.no_grad():
        tq = tqdm.tqdm(total=(len(tst_loader) * tst_loader.batch_size))
        tq.set_description(f'(test) Ep{epoch:>3d}')
        for im1,im2, name in tst_loader:
            tq.update(im1.size(0))
            input1 = im1.cuda()
            input2 = im2.cuda()
            outputs = model(input1, input2)
            y_pred_sigmoid = np.clip(torch.sigmoid(
                torch.squeeze(outputs)
            ).detach().cpu().numpy(), 0.0, 1.0)
            
            y_pred_mat = ss.csr_matrix(
                np.round(y_pred_sigmoid * 255).astype(np.uint8))
            ss.save_npz(
                str(Path(pred_mask_dir) / Path(f'{name[0]}.npz')),
                y_pred_mat)
        tq.close()
def make_sub(model_names, image_ids):  # noqa: C901
    chip_summary_list = []
    with tempfile.TemporaryDirectory() as tempdir:
        tq = tqdm.tqdm(total=(len(image_ids)))
        tq.set_description(f'(avgfolds)')
        for name in image_ids:
            tq.update(1)
            y_pred_avg = np.zeros((512, 512), dtype=np.float32)

            
            for model_name in model_names:
                # Prediction mask
                prefix = '_'.join(model_name.split('_')[:2])
                pred_mask_dir = f'/shared_ssd/gaohan/SenseChange/pred/mask_pred/test_{model_name}/'
                y_pred = np.array(ss.load_npz(
                    str(Path(pred_mask_dir) / Path(f'{name}.npz'))
                ).todense() / 255.0)
                y_pred_avg += y_pred
            y_pred_avg /= len(model_names)

            # Remove small objects
            y_pred = (y_pred_avg > 0.5)
            y_pred_label = skimage.measure.label(y_pred)
            preds_test = (y_pred_label > 0).astype('uint8')
            cv2.imwrite(f'/shared_ssd/gaohan/SenseChange/pred/mask_pred/merge/{name}.png',
                preds_test)


        tq.close()
        


def systemcheck_inference():

    assert helper_assertion_check("Check CUDA device is available",
                                  torch.cuda.is_available())


def systemcheck_train():
    assert helper_assertion_check("Check CUDA device is available",
                                  torch.cuda.is_available())
    assert helper_assertion_check("Check CUDA device count == 4",
                                  torch.cuda.device_count() == 4)
def helper_assertion_check(msg, res, max_length=80):
    print(msg, end='')
    if len(msg) > max_length - 6:
        print('\t', end='')
    else:
        space_size = max_length - 6 - len(msg)
        print(space_size * ' ', end='')

    if res:
        print('[ \x1b[6;32;40m' + 'OK' + '\x1b[0m ]')
        return True
    else:
        print('[ \x1b[6;31;40m' + 'NG' + '\x1b[0m ]')
        return False

if __name__ == "__main__":
    cli()