import os

if __name__ == '__main__':
    # os.system('/home/user/software/anaconda/envs/py37t11/bin/python -m torch.distributed.launch --master_port=9900 --nproc_per_node=1 train.py')
    # os.system('/home/hadoop-wallemnl/cephfs/data/shuchang/envs/py37t11/bin/python -m torch.distributed.launch --master_port=9900 --nproc_per_node=8 train.py')
    os.system(
        '/home/user/software/anaconda/envs/py37t11/bin/python -m torch.distributed.launch --master_port=9900 --nproc_per_node=1 train.py --config /home/user/Documents/code/fm_depth/config/cfg_kitti_fm_joint.py --work_dir /media/user/harddisk/weight/fmdepth')