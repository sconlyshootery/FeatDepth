import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import torch

sys.path.append('.')
from mono.model.mono_autoencoder.encoder import Encoder
from torchvision import transforms
from PIL import Image
use_pca = False


def show_ori_image(store_path, xx, yy):
    store_path = os.path.join(store_path, 'img')
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    with torch.no_grad():
        to_tensor = transforms.ToTensor()
        resize = transforms.Resize((256, 800), interpolation=Image.ANTIALIAS)
        img1 = pil_loader(os.path.join('/node01_data5/monodepth2-test/analysis/test', 'leftimg.jpg'))
        img1 = resize(img1)
        img1 = to_tensor(img1)
        img1 = img1.cuda()

        newf = img1.transpose(0,1).transpose(1,2).cpu().numpy()  # h,w
        h, w, _ = newf.shape
        # xx = [190, 335, 760]
        # yy = [100, 195, 180]
        fig = plt.figure()
        colors = ['red', 'red', 'red']
        for k in range(3):
            x, y = int(xx[k] / 256 * h), int(yy[k] / 800 * w)
            plt.axis('off')
            vmax = np.percentile(newf, 95)
            vmin = np.percentile(newf, 5)
            plt.imshow(newf, cmap='jet', vmax=vmax, vmin=vmin)
            plt.plot(x, y, marker='o', color=colors[k], linewidth=0.1)
        fig.savefig(os.path.join(store_path, "img.jpg"), bbox_inches='tight')
        plt.close(fig)

        for c in range(3):
            newf = img1.squeeze()[c].cpu().numpy()  # h,w
            h, w = newf.shape
            fig = plt.figure()
            for k in range(3):
                x,y = int(xx[k] / 256 * h),int(yy[k] / 800 * w)
                plt.axis('off')
                vmax = np.percentile(newf, 95)
                vmin = np.percentile(newf, 5)
                plt.imshow(newf, cmap='jet', vmax=vmax, vmin=vmin)
                plt.plot(x, y, marker='o',color=colors[k], linewidth=0.1)
            if not os.path.exists(os.path.join(store_path, str(c))):
                os.mkdir(os.path.join(store_path, str(c)))
            fig.savefig(os.path.join(store_path, str(c), "img.jpg"), bbox_inches='tight')
            plt.close(fig)

            for k in range(3):
                x, y = int(xx[k] / 256 * h), int(yy[k] / 800 * w)
                dx, dy = 10, 10
                img_crop = newf[ y- dy:y + dy,  x- dx:x + dx]
                img_crop = normalize(img_crop)
                saveimage(os.path.join(store_path, str(c), 'img_crop_{}_{}.jpg'.format(x,y)), img_crop,None,None)
    print('done!')


def build_extractor(num_layers, pretrained_path, store_path, xx, yy):
    extractor = Encoder(num_layers, '/node01/jobs/io/pretrained/checkpoints/resnet/resnet{}.pth'.format(num_layers))
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        for name, param in extractor.state_dict().items():
            extractor.state_dict()[name].copy_(checkpoint['state_dict']['Encoder.' + name])
    for param in extractor.parameters():
        param.requires_grad = False

    extractor.cuda()
    extractor.eval()
    with torch.no_grad():
        to_tensor = transforms.ToTensor()
        resize = transforms.Resize((256, 800), interpolation=Image.ANTIALIAS)
        img1 = pil_loader(os.path.join('/node01_data5/monodepth2-test/analysis/test', 'leftimg.jpg'))
        img1 = resize(img1)
        img1 = to_tensor(img1)
        img1 = img1.unsqueeze(0).cuda()

        """
        newf = extractor(img1)[0].squeeze().max(0)[0].cpu().numpy()#h,w
        """

        for c in range(64):
            newf = extractor(img1)[0].squeeze()[c].cpu().numpy()  # h,w
            h, w = newf.shape
            # xx = [190,335,760]
            # yy = [100,195,180]
            # xx = [80,335,760]
            # yy = [100,220,180]
            fig = plt.figure()
            colors=['red','red','red']
            for k in range(3):
                x,y = int(xx[k] / 256 * h),int(yy[k] / 800 * w)
                plt.axis('off')
                vmax = np.percentile(newf, 95)
                vmin = np.percentile(newf, 5)
                plt.imshow(newf, cmap='jet', vmax=vmax, vmin=vmin)
                plt.plot(x, y, marker='o',color=colors[k], linewidth=0.1)
            if not os.path.exists(os.path.join(store_path, str(c))):
                os.mkdir(os.path.join(store_path, str(c)))
            fig.savefig(os.path.join(store_path, str(c), "feature.jpg"), bbox_inches='tight')
            plt.close(fig)

            for k in range(3):
                x, y = int(xx[k] / 256 * h), int(yy[k] / 800 * w)
                dx, dy = 10, 10
                img_crop = newf[ y- dy:y + dy,  x- dx:x + dx]
                img_crop = normalize(img_crop)
                saveimage(os.path.join(store_path, str(c), 'img_crop_{}_{}.jpg'.format(xx[k],yy[k])), img_crop,None,None)
    print('done!')


def normalize(img):
    min = np.min(img)
    max = np.max(img)
    return (img-min)/(max-min+0.001)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


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


if __name__ == "__main__":
    store_path = '/node01_data5/monodepth2-test/analysis/fig3'
    if not os.path.exists(store_path):
        os.mkdir(store_path)

    xx = [60, 420, 760]
    yy = [100, 235, 180]

    model_path = [
    '/node01/jobs/io/out/changshu/autoencoder_0.001_0.01/epoch_30.pth',
    '/node01_data5/monodepth2-test/model/wow_ablation_d_1/epoch_30.pth',
    '/node01/jobs/io/out/changshu/autoencoder3/epoch_30.pth',
    ]

    show_ori_image(store_path, xx, yy)

    for k in range(len(model_path)):
        if not os.path.exists(os.path.join(store_path,str(k))):
            os.mkdir(os.path.join(store_path,str(k)))
        build_extractor(50, model_path[k], os.path.join(store_path,str(k)), xx, yy)