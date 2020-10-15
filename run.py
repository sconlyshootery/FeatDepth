import os

if __name__ == '__main__':
    # os.system('/home/user/software/anaconda/envs/py37t11/bin/python -m torch.distributed.launch --master_port=9900 --nproc_per_node=1 train.py')
    # os.system('/home/hadoop-wallemnl/cephfs/data/shuchang/envs/py37t11/bin/python -m torch.distributed.launch --master_port=9900 --nproc_per_node=8 train.py')
    os.system('/home/sconly/Documents/code/py37t11/bin/python -m torch.distributed.launch --master_port=9900 --nproc_per_node=1 train.py --config ./config/cfg_kitti_fm.py --work_dir /media/sconly/24eda5d5-e79b-423b-8dcc-8339a15f3219/weight/fmdepth')
    