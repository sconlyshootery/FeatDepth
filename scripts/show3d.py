import os
import sys

sys.path.append('.')
from mono.vis.mono_infer import MonoInfer

if __name__ == '__main__':
    img_dir = '/ssd/ETH3D/slam/cables_4/rgb'
    out_path = '/node01_data5/monodepth2-test/3D'
    cfg = {
        'model_name': 'Baseline',
        'num_layer' : 50,
        'input_shape' : (448, 736),
        'scale': 1/89.888, #1/5.4
        'model_path': '/node01/jobs/io/out/changshu/baseline_eth3d_ms/epoch_25.pth'
    }

    infer = MonoInfer(cfg)

    imgs = os.listdir(img_dir)
    for img in sorted(imgs):
        print('img is ', img)
        img_path = os.path.join(img_dir, img)
        infer.vis_result(img_path, out_path)





