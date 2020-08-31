DEPTH_LAYERS = 50
POSE_LAYERS = 18
FRAME_IDS = [0, -1, 1, 's']
IMGS_PER_GPU = 2
HEIGHT = 320
WIDTH = 1024

data = dict(
    name = 'kitti',
    split = 'test',#the split contains the list of testing data
    height = HEIGHT,
    width = WIDTH,
    frame_ids = FRAME_IDS,
    in_path = '/node01_data5/kitti_raw',#path to kitti raw data
    gt_depth_path = '/node01_data5/monodepth2-test/monodepth2/gt_depths.npz',#path to kitti depth ground truth
    png = False,
    stereo_scale=True if 's' in FRAME_IDS else False,
)

model = dict(
    name = 'mono_fm',
    depth_num_layers = DEPTH_LAYERS,
    pose_num_layers = POSE_LAYERS,
    frame_ids = FRAME_IDS,
    imgs_per_gpu = IMGS_PER_GPU,
    height = HEIGHT,
    width = WIDTH,
    scales = [0, 1, 2, 3],
    min_depth = 0.1,
    max_depth = 100.0,
    depth_pretrained_path = '/node01/jobs/io/pretrained/checkpoints/resnet/resnet{}.pth'.format(DEPTH_LAYERS),#path to pre-trained resnet weights
    pose_pretrained_path =  '/node01/jobs/io/pretrained/checkpoints/resnet/resnet{}.pth'.format(POSE_LAYERS),#path to pre-trained resnet weights
    extractor_pretrained_path = '/node01/jobs/io/out/changshu/autoencoder3/epoch_30.pth',
    automask=False if 's' in FRAME_IDS else True,
    disp_norm=False if 's' in FRAME_IDS else True,
    perception_weight=1e-3,
    smoothness_weight=1e-3,
)

#path to the weights trained on the kitti raw data training split
resume_from = '/node01_data5/monodepth2-test/model/wow_320_1024/epoch_40.pth'#we will resume from current epoch for further online refinement
total_epochs = 60# this value must be bigger than the epochs of the weight you resume from
#for example, you have trained 40 epoches on kitti raw data, and use this weight for resuming.
#When resuming, the program will start from epoch 41 and finish the rest of epoches (total_epochs - 40)
imgs_per_gpu = IMGS_PER_GPU
learning_rate = 1e-4
workers_per_gpu = 4
validate = True

optimizer = dict(type='Adam', lr=learning_rate, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[50],
    gamma=0.5,
)

checkpoint_config = dict(interval=1)
log_config = dict(interval=5,
                  hooks=[dict(type='TextLoggerHook'),])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
workflow = [('train', 1)]