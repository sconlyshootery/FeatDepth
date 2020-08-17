from __future__ import print_function
import os
import cv2
import math
import glob
import time
import json
import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

from utils import preprocess
from models.res_stereo import Res_Stereo
from utils import disp2color


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class DisparityPredictor(object):
    def __init__(self, model_path):
        self.model = Res_Stereo(4)
        self.model = nn.DataParallel(self.model, device_ids=[0])
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict['state_dict'])

        self.model.cuda()
        self.model.eval()
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in self.model.parameters()])))

    def image_64x(self, left_img, right_img):
        divisor = 64.
        self.H = left_img.shape[0]
        self.W = left_img.shape[1]
        self.H_ = int(math.ceil(self.H / divisor) * divisor)
        self.W_ = int(math.ceil(self.W / divisor) * divisor)
        imgL = cv2.resize(left_img, (self.W_, self.H_))
        imgR = cv2.resize(right_img, (self.W_, self.H_))
        imgL = processed(imgL).numpy()
        imgR = processed(imgR).numpy()
        imgL = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
        imgR = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])

        imgL_cuda = torch.FloatTensor(imgL).cuda()
        imgR_cuda = torch.FloatTensor(imgR).cuda()
        return imgL_cuda, imgR_cuda

    def disparity_restore(self, pred_disp):
        pred_disp = pred_disp * 20
        pred_disp = cv2.resize(pred_disp, (self.W, self.H)) * (self.W / float(self.W_))
        return pred_disp

    def inference(self, left_img, right_img):
        imgL_cuda, imgR_cuda = self.image_64x(left_img, right_img)

        start = time.time()
        with torch.no_grad():
            output = self.model(imgL_cuda, imgR_cuda)
        torch.cuda.synchronize()
        print('Speed: {:.2f} fps'.format(1.0 / (time.time() - start)))
        output = torch.squeeze(output)
        pred_disp = output.data.cpu().numpy()
        pred_disp = self.disparity_restore(pred_disp)
        return pred_disp


class DisparityVerifier(object):
    disp = None
    vis = None
    canvas = None
    h = None
    w = None
    pre = None
    end = (0, 0)

    left_click = False

    def __init__(self, disp_predictor):
        self.disp_predictor = disp_predictor

    @staticmethod
    def draw_frame(x, y):
        pt0 = (x, 0)
        pt1 = (x, 2 * DisparityVerifier.h)
        cv2.line(DisparityVerifier.canvas, pt0, pt1, color=(0, 0, 255), thickness=1)

        pt0 = (0, y)
        pt1 = (DisparityVerifier.w, y)
        cv2.line(DisparityVerifier.canvas, pt0, pt1, color=(0, 0, 255), thickness=1)

        pt0 = (0, DisparityVerifier.h + y)
        pt1 = (DisparityVerifier.w, DisparityVerifier.h + y)
        cv2.line(DisparityVerifier.canvas, pt0, pt1, color=(0, 0, 255), thickness=1)

    @staticmethod
    def mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('label:{}'.format(x))
            if not DisparityVerifier.left_click:
                DisparityVerifier.left_click = True
                pos = (x, y)
                cv2.circle(DisparityVerifier.vis, pos, 2, color=(0, 100, 255), thickness=-1)

                DisparityVerifier.draw_frame(x, y)
            else:
                DisparityVerifier.left_click = False
                pos = (x, y)
                cv2.circle(DisparityVerifier.vis, pos, 2, color=(0, 100, 255), thickness=-1)

        if event == cv2.EVENT_MOUSEMOVE:
            if not DisparityVerifier.left_click:

                DisparityVerifier.canvas[:, :, :] = DisparityVerifier.vis[:, :, :]

                if y < DisparityVerifier.h:
                    depth = '{:.2f} m'.format(0.54*721/DisparityVerifier.disp[y, x])
                    corr_pt = (x - DisparityVerifier.disp[y, x], y + DisparityVerifier.h)
                    cv2.putText(DisparityVerifier.canvas, depth, (corr_pt[0], corr_pt[1]-10), cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6,
                                color=(250, 242, 24), thickness=1, lineType=cv2.LINE_AA)

                    cv2.circle(DisparityVerifier.canvas, corr_pt, 4, color=(250, 242, 24), thickness=-1)

                DisparityVerifier.draw_frame(x, y)

    def verify(self, left_img, right_img):
        DisparityVerifier.disp = self.disp_predictor.inference(left_img, right_img).astype('uint16')

        DisparityVerifier.h, DisparityVerifier.w = left_img.shape[0:2]
        DisparityVerifier.vis = np.concatenate((left_img, right_img), axis=0)
        DisparityVerifier.canvas = DisparityVerifier.vis.copy()
        cv2.namedWindow('Verifier', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Verifier', DisparityVerifier.mouse_click)

        while True:
            cv2.imshow('Verifier', DisparityVerifier.canvas)
            k = cv2.waitKey(1)
            if k == ord('m'):
                mode = not mode
            elif k == ord('q'):
                break
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepStereo')
    parser.add_argument('--KITTI', default='2015',
                        help='KITTI version')
    parser.add_argument('--datapath', default='/kun/data2/KITTI/testing/',
                        help='select model')
    parser.add_argument('--loadmodel', default=None,
                        help='loading model')
    parser.add_argument('--feature_mode', default='PSP',
                        help='mode of feature extractor')
    parser.add_argument('--maxdisp', type=int, default=192,
                        help='maxium disparity')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    configs = json.load(open('config.json'))

    # test on kitti
    # test_examples_path = '/D/KITTI/object/testing/'
    # left_img_files = sorted(glob.glob(os.path.join(test_examples_path, 'image_2/*.png')))
    # right_img_files = sorted(glob.glob(os.path.join(test_examples_path, 'image_3/*.png')))

    # test on our own tri-camera system
    test_examples_path = '/home/kuiyuan/nas2/huaweiRosba/decoded/001/'
    left_img_files = sorted(glob.glob(os.path.join(test_examples_path, 'rect_img_left/*.png')))
    right_img_files = sorted(glob.glob(os.path.join(test_examples_path, 'rect_img_right/*.png')))

    processed = preprocess.get_transform(augment=False)

    disp_predictor = DisparityPredictor(args.loadmodel)

    h_crop = 1080-384
    w_crop = 1920-1248
    roi_top = h_crop/2
    roi_bottom = 1080-roi_top
    roi_left = w_crop/2
    roi_right = 1920-roi_left
    if True:
        disp_verifier = DisparityVerifier(disp_predictor)
        for inx in range(len(left_img_files)):

            left_img = cv2.imread(left_img_files[inx])
            right_img = cv2.imread(right_img_files[inx])

            left_img = left_img[roi_top:roi_bottom, roi_left:roi_right, :]
            right_img = right_img[roi_top:roi_bottom, roi_left:roi_right, :]

            # left_img = cv2.resize(left_img, (1248, 384))
            # right_img = cv2.resize(right_img, (1248, 384))

            disp_verifier.verify(left_img, right_img)


    for inx in range(len(left_img_files)):
        left_img = cv2.imread(left_img_files[inx])
        right_img = cv2.imread(right_img_files[inx])

        left_img = left_img[roi_top:roi_bottom, roi_left:roi_right, :]
        right_img = right_img[roi_top:roi_bottom, roi_left:roi_right, :]

        # left_img = cv2.resize(left_img, (1248, 384))
        # right_img = cv2.resize(right_img, (1248, 384))

        pred_disp = disp_predictor.inference(left_img, right_img)
        print('Max:{}, Min:{}'.format(pred_disp.max(), pred_disp.min()))

        # im_disp = (pred_disp * 256).astype('uint16')
        im_disp = cv2.applyColorMap(pred_disp.astype('uint8'), cv2.COLORMAP_HOT)

        im_disp = np.array(disp2color.disp_to_color(pred_disp*192)).astype(np.uint8)
        im_disp = cv2.cvtColor(im_disp, cv2.COLOR_BGR2RGB)

        cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
        cv2.imshow('Disparity', np.concatenate((left_img, im_disp), axis=0))
        cv2.waitKey(0)


