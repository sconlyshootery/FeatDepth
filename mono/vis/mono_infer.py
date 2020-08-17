import numpy as np
import torch
import cv2
from PIL import Image
import os

from .calibration import KittiCalibration
from .projection import Projector

from ..model.mono_baseline.depth_encoder import DepthEncoder
from ..model.mono_baseline.depth_decoder import DepthDecoder

class MonoInfer(object):
    def __init__(self, cfg):
        num_layer = cfg['num_layer']
        model_path = cfg['model_path']
        self.input_shape = cfg['input_shape']
        self.scale = cfg['scale']

        depth_encoder = DepthEncoder(num_layer, None)
        depth_decoder = DepthDecoder(depth_encoder.num_ch_enc)

        checkpoint = torch.load(model_path)
        for name, param in depth_encoder.state_dict().items():
            depth_encoder.state_dict()[name].copy_(checkpoint['state_dict']['DepthEncoder.' + name])
        for name, param in depth_decoder.state_dict().items():
            depth_decoder.state_dict()[name].copy_(checkpoint['state_dict']['DepthDecoder.' + name])

        depth_encoder.cuda()
        self.depth_encoder = depth_encoder.eval()
        depth_decoder.cuda()
        self.depth_decoder = depth_decoder.eval()


        self.calib = KittiCalibration('mono/datasets/splits/visual/calib.txt')
        # self.calib = AVPCalibration.from_params([1439.42968109, 1442.6983614, 608.3636072,319.89903815])
        self.pro_bev = Projector(self.calib)

    def __transform(self, cv2_img):
        im_tensor = torch.from_numpy(cv2_img.astype(np.float32)).cuda().unsqueeze(0)
        im_tensor = im_tensor.permute(0, 3, 1, 2).contiguous()
        im_tensor = torch.nn.functional.interpolate(im_tensor, self.input_shape, mode='bilinear', align_corners=False)
        im_tensor /= 255
        return im_tensor

    def predict(self, cv2_img):
        """
        :param img: str or numpy array
        :return:
        """
        original_height, original_width = cv2_img.shape[:2]
        im_tensor = self.__transform(cv2_img)

        with torch.no_grad():
            outputs = self.depth_decoder(self.depth_encoder(im_tensor))

        disp = outputs[("disp", 0, 0)]
        disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width),
                                                       mode="bilinear", align_corners=False)

        disp_resized_np = disp_resized.squeeze().cpu().numpy() * original_width * self.scale

        return disp_resized_np

    def vis_result(self, img_path, out_path):
        cv2_img = cv2.imread(img_path)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

        disp_resized_np = self.predict(cv2_img)
        bev = self.pro_bev.convert(disp_resized_np, cv2_img)
        bev = Image.fromarray(bev)
        save_name = os.path.basename(img_path)
        bev.save(os.path.join(out_path, save_name))



