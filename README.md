# feature_metric_depth
This is offical codes for the methods described in
> **Feature-metric Loss for Self-supervised Learning of Depth and Egomotion**
>
> [ECCV 2020](https://arxiv.org/pdf/2007.10603.pdf)

<p align="center">
  <img src="assets/p.png" alt="performance" width="600" />
</p>

If you find our work useful in your research please consider citing our paper:

```
@inproceedings{shu2020featdepth,
  title={Feature-metric Loss for Self-supervised Learning of Depth and Egomotion},
  author={Shu, Chang and Yu, Kun and Duan, Zhixiang and Yang, Kuiyuan},
  booktitle={ECCV},
  year={2020}
}
```

## Setup

### Requirements:
- PyTorch1.1+, Python3.5+, Cuda10.0+
- mmcv==0.4.4

Our codes are based on mmcv for distributed learning.
To make it convenient for you to train and test our codes, we provide our [anaconda environment](https://drive.google.com/file/d/1NSoGxhP8UpyW-whzpqP3WIB6u2mgGP49/view?usp=sharing), 
you only need to download it and extract it to the folder of your anaconda environments, and use the python in it to run our codes.

If you would like to set up your anaconda environment by yourself, you can do as follows:
```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name featdepth python=3.7
conda activate featdepth

# this installs the right pip and dependencies for the fresh python
conda install ipython
conda install pip

# install required packages from requirements.txt
pip install -r requirements.txt
```

## KITTI training data

Our training data is the same with other self-supervised monocular depth estimation methods, please refer to [monodepth2](https://github.com/nianticlabs/monodepth2) to prepare the training data.

## pretrained weights

We provide weights for:  
(1) [AutoEncoder trained on the kitti raw data](https://drive.google.com/file/d/1ncAWUMvLq2ETMpG-7eI9qfILce_cPPfy/view?usp=sharing);  
(2) [FeatDepth trained on the kitti raw data](https://drive.google.com/file/d/1HlAubfuja5nBKpfNU3fQs-3m3Zaiu9RI/view?usp=sharing);  
(3) [FeatDepth finetuned on the test split of kitti raw data by using online refinement](https://drive.google.com/file/d/1CfCtz55s4QHya3y3UslxsuD_0cxNlA-D/view?usp=sharing);  
(4) [FeatDepth trained on kitti odometry](https://drive.google.com/file/d/1vQJbiyPXv_XNQYpyVocDB3-LKwx2LVka/view?usp=sharing);  
(5) [FeatDepth trained on Euroc](https://drive.google.com/file/d/1IMIAKpHXmqyUxiUIiqqp5qI-nJXDUSmj/view?usp=sharing);  
(6) [FeatDepth trained on NYU](https://drive.google.com/file/d/1Mo050P-DgG-jrNXWww07GXXyst5h5Q74/view?usp=sharing).

## API
We provide an API interface for you to predict depth and pose from an image sequence and visulize some results.
They are stored in folder 'scripts'.
```
draw_odometry.py is used to provide several analytical curves and obtain standard kitti odometry evaluation results.
```

```
eval_pose.py is used to obtain kitti odometry evaluation results.
```

```
eval_depth.py is used to obtain kitti depth evaluation results.
```

```
infer.py is used to generate depth maps from given models.
```

```
infer_singleimage.py is used to test a single image for view.
```
## Training
You can use following command to launch distributed learning of our model:
```shell
/path/to/python -m torch.distributed.launch --master_port=9900 --nproc_per_node=1 train.py --config /path/to/cfg_kitti_fm.py --work_dir /dir/for/saving/weights/and/logs'
```
Here nproc_per_node refers to GPU number you want to use.

## Configurations
We provide a variety of config files for training on different datasets.
They are stored in config folder.

For example:  
(1) 'cfg_kitti_fm.py' is used to train our model on kitti dataset, where the weights of autoencoder are loaded from the pretrained weights we provide and fixed during the traing. 
This mode is prefered when your GPU memory is lower than 16 GB;  
(2) 'cfg_kitti_fm_joint.py' is used to train our model on kitti dataset, where the autoencoder is jointly trained with depthnet and posenet.
We rescale the input resolution of our model to ensure training with 12 GB GPU memory, slightly reducing the performance.
You can modify the input resolution according to your computational resource.

For modifying config files, please refer to cfg_kitti_fm.py.

## Online refinement
We provide cfg file for online refinement, you can use cfg_kitti_fm_refine.py to refine your model trained on kitti raw data by keeping training on test data.
For settings of online refinement, please refer to details in cfg_kitti_fm_refine.py in the folder config.

## Finetuning
If you want to finetune on a given weights, you can modify the 'finetune' term from 'None' to an existing path to a pre-trained weight in the config files.

## Resuming
If you want to reproduce the training state of a certain pretrained weight, you can modify the 'resume_from' term from 'None' to an existing path to a pre-trained weight in the config files.
The program will continue training from where the pretrained weight ends.
Note that you have to increase the 'total_epochs' value to make sure that the training have enough epochs left to continue.

## Notes
Our model predicts inverse depths.
If you want to get real depth when training stereo model, you have to convert inverse depth to depth, and then multiply it by 36.
