import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import torch
from torch.utils.data import DataLoader

sys.path.append('.')
from mono.datasets.kitti_dataset import KITTIRAWDataset
from mono.datasets.utils import readlines
from mono.model.mono_autoencoder.encoder import Encoder

use_pca = True

def build_extractor(num_layers, pretrained_path):
    extractor = Encoder(num_layers, None)
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    for name, param in extractor.state_dict().items():
        extractor.state_dict()[name].copy_(checkpoint['state_dict']['Encoder.' + name])
    for param in extractor.parameters():
        param.requires_grad = False
    filenames = ['2011_09_26/2011_09_26_drive_0002_sync 0000000069 l', '2011_09_26/2011_09_26_drive_0002_sync 0000000068 l']
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
    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()
            if use_pca:
                f = extractor(inputs[("color", 0, 0)])[0].squeeze().cpu().numpy()  # c,h,w
                c,h,w = f.shape
                f = f.reshape(c, -1)  # c,h*w
                pca = PCA(n_components=3)
                pca_f = pca.fit_transform(f.T)#h*w,3
                new_f = pca_f.transpose(1, 0)#3,h*w

                A = [[1, 1, 1], [1, -1, 0], [.5, .5, -1]]
                A = np.linalg.inv(A)
                new_f = np.matmul(A, new_f)

                max = np.max(new_f)
                min = np.min(new_f)
                new_f = new_f - min
                new_f = new_f / max
                newf = new_f.reshape(3,h,w).transpose((1,2,0)).clip(0,1)
            else:
                newf = extractor(inputs[("color", 0, 0)])[0].squeeze().max(0)[0].cpu().numpy()#h,w

            saveimage(os.path.join('/node01_data5/monodepth2-test/analysis/fig3', str(batch_idx) + ".jpg"), newf)
    print('done!')

def saveimage(imgname,img):
#     vmax = np.percentile(img, 95)
#     vmin = np.percentile(img, 5)
    plt.imsave(imgname, img, cmap='jet')


if __name__ == "__main__":
    build_extractor(50, '/node01/jobs/io/out/changshu/autoencoder3/epoch_30.pth')