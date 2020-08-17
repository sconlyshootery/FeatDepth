import matplotlib.pyplot as plt
import os
import numpy as np

data_path = '/node01_data5/monodepth2-test/analysis/img'


def gradient(D):
    h,w,c = D.shape
    dy = np.abs(D[1:, :] - D[:-1,:])
    dx = np.abs(D[:, 1:] - D[:, :-1])
    dx = np.resize(dx, (h, w, c))
    dy = np.resize(dy, (h, w, c))
    return dx, dy

names = sorted(os.listdir(data_path))
for i in range(len(names)):
    img = plt.imread(os.path.join(data_path, names[i]))
    img = np.array(img)
    dx, dy = gradient(img)
    dx = np.linalg.norm(dx, ord=2, axis=2)
    dy = np.linalg.norm(dy, ord=2, axis=2)
    # energy_map = dx+dy
    energy_map = (np.exp(-dx) + np.exp(-dy)) / 2
    # max = np.amax(energy_map)
    # min = np.amin(energy_map)
    # energy_map = (energy_map-min)/(max-min)

    vmax = np.percentile(energy_map, 95)
    plt.imsave('/node01_data5/monodepth2-test/analysis/energy/{}.jpg'.format(i), (energy_map*255).astype(np.uint8), cmap='magma', vmax=vmax)

print("done!")

