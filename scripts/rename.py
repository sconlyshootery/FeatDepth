import os

data_path = '/node01_data5/monodepth2-test/kitti_test'
f1 = os.path.join(data_path, 'mono_1024_320')
f2 = os.path.join(data_path, 'mono+stereo_1024_320')

# names = os.listdir(f1)
# for i in range(len(names)):
#     os.rename('/home/sconly/Documents/kitti_test/mono_640_192/{}.jpg'.format(i), '/home/sconly/Documents/kitti_test/mono_640_192/{:04d}.jpg'.format(i))


names = os.listdir(f2)
for i in range(len(names)):
    os.rename(f2+'/{}.jpg'.format(i), f2+'/{:04d}.jpg'.format(i))

