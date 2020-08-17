from __future__ import absolute_import, division, print_function
import torch
import torch.nn.functional as F
import torch.nn as nn

from .layers import SSIM, Backproject, Project
from .depth_encoder import DepthEncoder
from .depth_decoder import DepthDecoder
from .pose_encoder import PoseEncoder
from .pose_decoder import PoseDecoder
from .encoder import Encoder
from .decoder import Decoder
from ..registry import MONO


@MONO.register_module
class mono_fm_joint(nn.Module):
    def __init__(self, options):
        super(mono_fm_joint, self).__init__()
        self.opt = options
        self.DepthEncoder = DepthEncoder(self.opt.depth_num_layers,
                                         self.opt.depth_pretrained_path)
        self.DepthDecoder = DepthDecoder(self.DepthEncoder.num_ch_enc)
        self.PoseEncoder = PoseEncoder(self.opt.pose_num_layers,
                                       self.opt.pose_pretrained_path)
        self.PoseDecoder = PoseDecoder(self.PoseEncoder.num_ch_enc)
        self.Encoder = Encoder(self.opt.depth_num_layers, self.opt.depth_pretrained_path)
        self.Decoder = Decoder(self.Encoder.num_ch_enc)
        self.ssim = SSIM()
        self.backproject = Backproject(self.opt.imgs_per_gpu, self.opt.height, self.opt.width)
        self.project= Project(self.opt.imgs_per_gpu, self.opt.height, self.opt.width)

    def forward(self, inputs):
        outputs = self.DepthDecoder(self.DepthEncoder(inputs["color_aug", 0, 0]))
        if self.training:
            outputs.update(self.predict_poses(inputs))
            features = self.Encoder(inputs[("color", 0, 0)])
            outputs.update(self.Decoder(features, 0))
            loss_dict = self.compute_losses(inputs, outputs, features)
            return outputs, loss_dict
        return outputs

    def robust_l1(self, pred, target):
        eps = 1e-3
        return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)

    def compute_perceptional_loss(self, tgt_f, src_f):
        loss = self.robust_l1(tgt_f, src_f).mean(1, True)
        return loss

    def compute_reprojection_loss(self, pred, target):
        photometric_loss = self.robust_l1(pred, target).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = (0.85 * ssim_loss + 0.15 * photometric_loss)
        return reprojection_loss

    def compute_losses(self, inputs, outputs, features):
        loss_dict = {}
        target = inputs[("color", 0, 0)]

        for i in range(5):
            f=features[i]
            regularization_loss = self.get_feature_regularization_loss(f, target)
            loss_dict[('feature_regularization_loss', i)] = regularization_loss/(2 ** i)/5

        for scale in self.opt.scales:
            """
            initialization
            """
            disp = outputs[("disp", 0, scale)]

            reprojection_losses = []
            perceptional_losses = []

            """
            autoencoder
            """
            res_img = outputs[("res_img", 0, scale)]
            _, _, h, w = res_img.size()
            target_resize = F.interpolate(target, [h, w], mode="bilinear", align_corners=False)
            img_reconstruct_loss = self.compute_reprojection_loss(res_img, target_resize)
            loss_dict[('img_reconstruct_loss', scale)] = img_reconstruct_loss.mean() / len(self.opt.scales)

            """
            reconstruction
            """
            outputs = self.generate_images_pred(inputs, outputs, scale)
            outputs = self.generate_features_pred(inputs, outputs)

            """
            automask
            """
            if self.opt.automask:
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, 0)]
                    identity_reprojection_loss = self.compute_reprojection_loss(pred, target)
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 1e-5
                    reprojection_losses.append(identity_reprojection_loss)

            """
            minimum reconstruction loss
            """
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_loss = torch.cat(reprojection_losses, 1)

            min_reconstruct_loss, outputs[("min_index", scale)] = torch.min(reprojection_loss, dim=1)
            loss_dict[('min_reconstruct_loss', scale)] = min_reconstruct_loss.mean()/len(self.opt.scales)

            """
            minimum perceptional loss
            """
            for frame_id in self.opt.frame_ids[1:]:
                src_f = outputs[("feature", frame_id, 0)]
                tgt_f = self.Encoder(inputs[("color", 0, 0)])[0]
                perceptional_losses.append(self.compute_perceptional_loss(tgt_f, src_f))
            perceptional_loss = torch.cat(perceptional_losses, 1)

            min_perceptional_loss, outputs[("min_index", scale)] = torch.min(perceptional_loss, dim=1)
            loss_dict[('min_perceptional_loss', scale)] = self.opt.perception_weight * min_perceptional_loss.mean() / len(self.opt.scales)

            """
            disp mean normalization
            """
            if self.opt.disp_norm:
                mean_disp = disp.mean(2, True).mean(3, True)
                disp = disp / (mean_disp + 1e-7)

            """
            smooth loss
            """
            smooth_loss = self.get_smooth_loss(disp, target)
            loss_dict[('smooth_loss', scale)] = self.opt.smoothness_weight * smooth_loss / (2 ** scale)/len(self.opt.scales)

        return loss_dict

    def disp_to_depth(self, disp, min_depth, max_depth):
        min_disp = 1 / max_depth  # 0.01
        max_disp = 1 / min_depth  # 10
        scaled_disp = min_disp + (max_disp - min_disp) * disp  # (10-0.01)*disp+0.01
        depth = 1 / scaled_disp
        return scaled_disp, depth

    def predict_poses(self, inputs):
        outputs = {}
        #[192,640] for kitti
        pose_feats = {f_i: F.interpolate(inputs["color_aug", f_i, 0], [192, 640], mode="bilinear", align_corners=False) for f_i in self.opt.frame_ids}
        for f_i in self.opt.frame_ids[1:]:
            if not f_i == "s":
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]
                pose_inputs = self.PoseEncoder(torch.cat(pose_inputs, 1))
                axisangle, translation = self.PoseDecoder(pose_inputs)
                outputs[("cam_T_cam", 0, f_i)] = self.transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        return outputs

    def generate_images_pred(self, inputs, outputs, scale):
        disp = outputs[("disp", 0, scale)]
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        _, depth = self.disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            if frame_id == "s":
                T = inputs["stereo_T"]
            else:
                T = outputs[("cam_T_cam", 0, frame_id)]
            cam_points = self.backproject(depth, inputs[("inv_K")])
            pix_coords = self.project(cam_points, inputs[("K")], T)#[b,h,w,2]
            img = inputs[("color", frame_id, 0)]
            outputs[("color", frame_id, scale)] = F.grid_sample(img, pix_coords, padding_mode="border")
        return outputs

    def generate_features_pred(self, inputs, outputs):
        disp = outputs[("disp", 0, 0)]
        disp = F.interpolate(disp, [int(self.opt.height/2), int(self.opt.width/2)], mode="bilinear", align_corners=False)
        _, depth = self.disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            if frame_id == "s":
                T = inputs["stereo_T"]
            else:
                T = outputs[("cam_T_cam", 0, frame_id)]

            backproject = Backproject(self.opt.imgs_per_gpu, int(self.opt.height/2), int(self.opt.width/2))
            project = Project(self.opt.imgs_per_gpu, int(self.opt.height/2), int(self.opt.width/2))

            cam_points = backproject(depth, inputs[("inv_K")])
            pix_coords = project(cam_points, inputs[("K")], T)#[b,h,w,2]
            img = inputs[("color", frame_id, 0)]
            src_f = self.Encoder(img)[0]
            outputs[("feature", frame_id, 0)] = F.grid_sample(src_f, pix_coords, padding_mode="border")
        return outputs

    def transformation_from_parameters(self, axisangle, translation, invert=False):
        R = self.rot_from_axisangle(axisangle)
        t = translation.clone()
        if invert:
            R = R.transpose(1, 2)
            t *= -1
        T = self.get_translation_matrix(t)
        if invert:
            M = torch.matmul(R, T)
        else:
            M = torch.matmul(T, R)
        return M

    def get_translation_matrix(self, translation_vector):
        T = torch.zeros(translation_vector.shape[0], 4, 4).cuda()
        t = translation_vector.contiguous().view(-1, 3, 1)
        T[:, 0, 0] = 1
        T[:, 1, 1] = 1
        T[:, 2, 2] = 1
        T[:, 3, 3] = 1
        T[:, :3, 3, None] = t
        return T

    def rot_from_axisangle(self, vec):
        angle = torch.norm(vec, 2, 2, True)
        axis = vec / (angle + 1e-7)
        ca = torch.cos(angle)
        sa = torch.sin(angle)
        C = 1 - ca
        x = axis[..., 0].unsqueeze(1)
        y = axis[..., 1].unsqueeze(1)
        z = axis[..., 2].unsqueeze(1)
        xs = x * sa
        ys = y * sa
        zs = z * sa
        xC = x * C
        yC = y * C
        zC = z * C
        xyC = x * yC
        yzC = y * zC
        zxC = z * xC
        rot = torch.zeros((vec.shape[0], 4, 4)).cuda()
        rot[:, 0, 0] = torch.squeeze(x * xC + ca)
        rot[:, 0, 1] = torch.squeeze(xyC - zs)
        rot[:, 0, 2] = torch.squeeze(zxC + ys)
        rot[:, 1, 0] = torch.squeeze(xyC + zs)
        rot[:, 1, 1] = torch.squeeze(y * yC + ca)
        rot[:, 1, 2] = torch.squeeze(yzC - xs)
        rot[:, 2, 0] = torch.squeeze(zxC - ys)
        rot[:, 2, 1] = torch.squeeze(yzC + xs)
        rot[:, 2, 2] = torch.squeeze(z * zC + ca)
        rot[:, 3, 3] = 1
        return rot

    def get_smooth_loss(self, disp, img):
        b, _, h, w = disp.size()
        a1 = 0.5
        a2 = 0.5
        img = F.interpolate(img, (h, w), mode='area')

        disp_dx, disp_dy = self.gradient(disp)
        img_dx, img_dy = self.gradient(img)

        disp_dxx, disp_dxy = self.gradient(disp_dx)
        disp_dyx, disp_dyy = self.gradient(disp_dy)

        img_dxx, img_dxy = self.gradient(img_dx)
        img_dyx, img_dyy = self.gradient(img_dy)

        smooth1 = torch.mean(disp_dx.abs() * torch.exp(-a1 * img_dx.abs().mean(1, True))) + \
                  torch.mean(disp_dy.abs() * torch.exp(-a1 * img_dy.abs().mean(1, True)))

        smooth2 = torch.mean(disp_dxx.abs() * torch.exp(-a2 * img_dxx.abs().mean(1, True))) + \
                  torch.mean(disp_dxy.abs() * torch.exp(-a2 * img_dxy.abs().mean(1, True))) + \
                  torch.mean(disp_dyx.abs() * torch.exp(-a2 * img_dyx.abs().mean(1, True))) + \
                  torch.mean(disp_dyy.abs() * torch.exp(-a2 * img_dyy.abs().mean(1, True)))

        return smooth1+smooth2

    def gradient(self, D):
        D_dy = D[:, :, 1:] - D[:, :, :-1]
        D_dx = D[:, :, :, 1:] - D[:, :, :, :-1]
        return D_dx, D_dy

    def get_feature_regularization_loss(self, feature, img):
        b, _, h, w = feature.size()
        img = F.interpolate(img, (h, w), mode='area')

        feature_dx, feature_dy = self.gradient(feature)
        img_dx, img_dy = self.gradient(img)

        feature_dxx, feature_dxy = self.gradient(feature_dx)
        feature_dyx, feature_dyy = self.gradient(feature_dy)

        img_dxx, img_dxy = self.gradient(img_dx)
        img_dyx, img_dyy = self.gradient(img_dy)

        smooth1 = torch.mean(feature_dx.abs() * torch.exp(-img_dx.abs().mean(1, True))) + \
                  torch.mean(feature_dy.abs() * torch.exp(-img_dy.abs().mean(1, True)))

        smooth2 = torch.mean(feature_dxx.abs() * torch.exp(-img_dxx.abs().mean(1, True))) + \
                  torch.mean(feature_dxy.abs() * torch.exp(-img_dxy.abs().mean(1, True))) + \
                  torch.mean(feature_dyx.abs() * torch.exp(-img_dyx.abs().mean(1, True))) + \
                  torch.mean(feature_dyy.abs() * torch.exp(-img_dyy.abs().mean(1, True)))

        return -self.opt.dis * smooth1+ self.opt.cvt * smooth2