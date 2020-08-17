import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

sys.path.append('.')
from mono.datasets.kitti_dataset import KITTIRAWDataset
from mono.model.mono_autoencoder.encoder import Encoder
from mono.model.mono_autoencoder.layers import SSIM


store_path = '/node01_data5/monodepth2-test/analysis/fig3'



def build_extractor(num_layers, pretrained_path):
    extractor = Encoder(num_layers, '/node01/jobs/io/pretrained/checkpoints/resnet/resnet{}.pth'.format(50))

    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        for name, param in extractor.state_dict().items():
            extractor.state_dict()[name].copy_(checkpoint['state_dict']['Encoder.' + name])

    for param in extractor.parameters():
        param.requires_grad = False
    return extractor


def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)


def saveimage(imgname,img,x,y):
    fig = plt.figure()
    plt.axis('off')
    vmax = np.percentile(img, 95)
    vmin = np.percentile(img, 5)
    plt.imshow(img, cmap='jet', vmax=vmax, vmin=vmin)
    if x is not None and y is not None:
        plt.plot(x, y, 'r.', linewidth=0.1)
    fig.savefig(imgname, bbox_inches='tight')
    plt.close(fig)


def visualize(x, y, k, pretrained_path):
    extractor = build_extractor(50, pretrained_path)

    filenames = ['2011_09_26/2011_09_26_drive_0046_sync 0000000085 l',
                 '2011_09_26/2011_09_26_drive_0046_sync 0000000085 r']

    dataset = KITTIRAWDataset('/node01_data5/kitti_raw',
                              filenames,
                              256,
                              800,
                              [0],
                              is_train=False,
                              gt_depth_path='/node01_data5/monodepth2-test/monodepth2/gt_depths.npz')

    dataloader = DataLoader(dataset,
                            1,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)
    extractor.cuda()
    extractor.eval()

    f_list = []
    img_list = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            f = extractor(inputs[("color", 0, 0)])[0].cpu().numpy()
            img = inputs[("color", 0, 0)].squeeze().transpose(0, 1).transpose(1, 2).cpu().numpy()
            f_list.append(f)
            img_list.append(img)

    f1 = np.squeeze(f_list[0])
    f2 = np.squeeze(f_list[1])
    img1 = img_list[0]
    img2 = img_list[1]
    p1 = f1[:, int(y / 2), int(x / 2), np.newaxis, np.newaxis]
    df = np.sum(np.abs(f2 - p1), axis=0)

    interval = 16
    zoom = df[int(y / 2), int(x / 2) - interval:int(x / 2) + interval]
    ind = np.argmin(zoom)
    xx = int(x / 2) - interval + ind

    saveimage(os.path.join(store_path, str(x)+"_"+str(y)+"_"+str(k) + "_img1.jpg"), img1, x, y)
    saveimage(os.path.join(store_path, str(x)+"_"+str(y)+"_"+str(k) + "_img2.jpg"), img2, xx*2, y)

    names=['ResNet50', 'Ours']
    plt.plot(range(-interval, interval), [(y - min(zoom)) / max(zoom) for y in zoom], linewidth=1, marker='o', label=names[k])
    print('match{}='.format(k), xx*2)
    print('y{}=['.format(k))
    for y in [(y - min(zoom)) / max(zoom) for y in zoom]:
        print(y, ',')
    print(']')


def pe(x, y, k):
    filenames = ['2011_09_26/2011_09_26_drive_0046_sync 0000000085 l',
                 '2011_09_26/2011_09_26_drive_0046_sync 0000000085 r']

    dataset = KITTIRAWDataset('/node01_data5/kitti_raw',
                              filenames,
                              256,
                              800,
                              [0],
                              is_train=False,
                              gt_depth_path='/node01_data5/monodepth2-test/monodepth2/gt_depths.npz')

    dataloader = DataLoader(dataset,
                            1,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)

    img_list = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            img = inputs[("color", 0, 0)].squeeze().transpose(0, 1).transpose(1, 2).cpu().numpy()
            img_list.append(img)

    img1 = img_list[0]
    img2 = img_list[1]
    p1 = img1[y, x, :]
    p1 = p1[np.newaxis, np.newaxis, :]
    df = np.sum(np.abs(img2 - p1), axis=2)
    print(df.shape)

    interval = 16
    zoom = df[y, x - interval:x + interval]
    ind = np.argmin(zoom)
    xx = x - interval + ind

    saveimage(os.path.join(store_path, str(x)+"_"+str(y)+"_"+str(k)+"_img1.jpg"),img1,x,y)
    saveimage(os.path.join(store_path, str(x)+"_"+str(y)+"_"+str(k)+"_img2.jpg"),img2,xx,y)
    plt.plot(range(-interval, interval), [(y - min(zoom)) / max(zoom) for y in zoom], linewidth=1, marker='o', label='pe')
    print('match{}='.format(k), xx)
    print('y{}=['.format(k))
    for y in [(y - min(zoom)) / max(zoom) for y in zoom]:
        print(y, ',')
    print(']')


def pe2(x, y, k):
    filenames = ['2011_09_26/2011_09_26_drive_0046_sync 0000000085 l',
                 '2011_09_26/2011_09_26_drive_0046_sync 0000000085 r']

    dataset = KITTIRAWDataset('/node01_data5/kitti_raw',
                              filenames,
                              256,
                              800,
                              [0],
                              is_train=False,
                              gt_depth_path='/node01_data5/monodepth2-test/monodepth2/gt_depths.npz')

    dataloader = DataLoader(dataset,
                            1,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)
    metric = SSIM()
    img_list = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            img = inputs[("color", 0, 0)]
            img_list.append(img)

    img1 = img_list[0]
    img2 = img_list[1]
    p1 = img1[:,:, y, x].unsqueeze(-1).unsqueeze(-1).expand_as(img1)
    df = 0.15*(p1-img2).abs()+0.85*metric(p1, img2)#1,c,h,w
    df = df.squeeze().sum(0).cpu().numpy()
    print(df.shape)

    interval = 16
    zoom = df[y, x - interval:x + interval]
    ind = np.argmin(zoom)
    xx = x - interval + ind

    saveimage(os.path.join(store_path, str(x)+"_"+str(y)+"_"+str(k)+"_img1.jpg"),img1.squeeze().transpose(0,1).transpose(1,2).cpu().numpy(),x,y)
    saveimage(os.path.join(store_path, str(x)+"_"+str(y)+"_"+str(k)+"_img2.jpg"),img2.squeeze().transpose(0,1).transpose(1,2).cpu().numpy(),xx,y)
    plt.plot(range(-interval, interval), [(y - min(zoom)) / max(zoom) for y in zoom], linewidth=1, marker='o', label='pe2')
    print('match{}='.format(k), xx)
    print('y{}=['.format(k))
    for y in [(y - min(zoom)) / max(zoom) for y in zoom]:
        print(y, ',')
    print(']')

if __name__ == "__main__":
    path_list = [None,
                 '/node01/jobs/io/out/changshu/autoencoder3/epoch_30.pth',
                ]

    x,y = 500, 180
    fig = plt.figure()
    for k in range(len(path_list)):
        visualize(x, y, k, path_list[k])
    pe(x, y, len(path_list))
    pe2(x, y, len(path_list)+1)
    plt.legend(fontsize=10)
    plt.show()
    fig.savefig(os.path.join(store_path, str(x) + "_" + str(y) + "_scatter.jpg"))
    print('done!')
