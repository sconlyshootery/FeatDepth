# fm_depth
This is the reference PyTorch implementation for training and testing depth and pose estimation models using the method described in

> **Feature-metric Loss for Self-supervised Learning of Depth and Egomotion**
>
> [Chang Shu], [Kun Yu], [Zhixiang Duan] and [Kuiyuan Yang]  
>
> [ECCV 2020]

This is our unsupervise monocular depth estimation model, whose performance is listed below.

<p align="center">
 <img src="assets/p.png" alt="performance results" width="600" />
</p>

## KITTI training data

Training data is stored in node1/data5/kitti_raw and deepmotion5/data2/raw.

## pretrained weights

Our pretrained weights are stored in node1/data5/monodepth-test/model.
There mainly are 3 kinds of models, that is m_baseline, s_baseline and ms_baseline, which respectively represent the model trained on mono, stereo or mono+stereo data.
All the models can predict depth from a single image.
For example, 'ms_baseline_50_256_800.pth' means this model is trained on mono+stere data and resnet50 is used as backbone and 256*800 image size is used.

## API
We provide an API interface for you to predict depth and pose from an image sequence and visulize some results.
They are stored in folder 'scripts'.

## Training
We provide a different configuration for different perferrence.
'cfg_kitti_baseline_m.py' is used to train baseline model on kitti dataset with mono data.
'cfg_kitti_baseline_m.spy' is used to train baseline model on kitti dataset with mono+stereo data.

## Finetuning
If you want to finetune on a given weights, we can modify the 'resume_from' term from 'None' to an existing path to a pre-trained weight.

## Notes
Our model predicts inverse depths.
If you want to get real depth when training stereo model, you have to convert inverse depth to depth, and then multiply it by 36.
