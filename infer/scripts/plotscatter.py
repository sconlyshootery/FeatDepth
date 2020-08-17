import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize



match0= 768
y0=[
0.11148792 ,
0.12518685 ,
0.14353137 ,
0.080361165 ,
0.12090467 ,
0.052681264 ,
0.061892737 ,
0.064098194 ,
0.08489604 ,
0.11487848 ,
0.24237756 ,
0.022375835 ,
0.28503385 ,
0.079363875 ,
0.30815727 ,
0.06674892 ,
0.07223882 ,
0.08627712 ,
0.18007833 ,
0.13361779 ,
0.0 ,
0.04176147 ,
0.084983215 ,
0.17804354 ,
0.10010588 ,
0.026450748 ,
0.24413237 ,
0.24729869 ,
0.034770954 ,
0.08123001 ,
0.2093575 ,
0.52419186 ,
]
match1= 760
y1=[
0.057935353 ,
0.0995803 ,
0.10192102 ,
0.057726324 ,
0.01330022 ,
0.060157947 ,
0.07459866 ,
0.09827158 ,
0.21143177 ,
0.27865285 ,
0.28849342 ,
0.25949112 ,
0.2481389 ,
0.16906299 ,
0.070587724 ,
0.011196518 ,
0.0 ,
0.028901616 ,
0.07998927 ,
0.063433714 ,
0.029182917 ,
0.0019150225 ,
0.06095949 ,
0.113518186 ,
0.12248458 ,
0.11357706 ,
0.1315784 ,
0.22890016 ,
0.31452972 ,
0.38438204 ,
0.4910386 ,
0.85058385 ,
]

match2= 757
y2=[
0.9166667 ,
0.6666667 ,
0.6666667 ,
0.9166667 ,
0.5833333 ,
0.41666666 ,
0.6666667 ,
0.6666667 ,
0.9166667 ,
0.41666666 ,
0.16666667 ,
0.6666667 ,
0.41666666 ,
0.0 ,
0.16666667 ,
0.41666666 ,
0.16666667 ,
0.083333336 ,
0.083333336 ,
0.5833333 ,
0.41666666 ,
0.16666667 ,
0.16666667 ,
0.16666667 ,
0.083333336 ,
0.083333336 ,
0.083333336 ,
0.16666667 ,
0.083333336 ,
0.33333334 ,
0.5 ,
0.33333334 ,
]

match3= 745
y3=[
0.13653629 ,
0.0 ,
0.15831336 ,
0.32417852 ,
0.6137119 ,
0.32333398 ,
0.17728171 ,
0.16422677 ,
0.38832498 ,
0.73101574 ,
0.4246274 ,
0.13580868 ,
0.58089757 ,
0.3877765 ,
0.38337553 ,
0.15471485 ,
0.21538736 ,
0.3183769 ,
0.36442357 ,
0.12415356 ,
0.46587536 ,
0.1951933 ,
0.13473809 ,
0.057891537 ,
0.10260951 ,
0.07651117 ,
0.16406833 ,
0.07190964 ,
0.023642741 ,
0.23763485 ,
0.33885342 ,
0.28942278 ,
]



fontsize = 15
x = range(-16,16)
i, j = 335, 195
fig=plt.figure()
plt.title('Loss Landscape' ,fontsize=fontsize)
plt.xlabel('Disparity',fontsize=fontsize)
plt.ylabel('Loss',fontsize=fontsize)

plt.plot(x, y0, linewidth=1, color='b', marker='.', label='ResNet')
plt.plot(x, y4, linewidth=1, color='y', marker='.',label='PSMNet')
plt.plot(x, y2, linewidth=1, color='g', marker='.',label='L1')
plt.plot(x, y3, linewidth=1, color='c', marker='.',label='L1+SSIM')
plt.plot(x, y1, linewidth=1, color='r', marker='.',label='Ours')

plt.ylim(-0.05, 0.8)
plt.plot(-9*np.ones(10), np.linspace(-.05,.4,10),color='black', linestyle="--", label='gt')
plt.legend(fontsize=10)
# plt.show()
fig.savefig('./test/landscape.png')
plt.close(fig)

img1 = plt.imread('./test/leftimg.jpg')
fig = plt.figure()
plt.axis('off')
plt.imshow(img1)
plt.plot(i, j, 'r.', linewidth=0.1)
fig.savefig('./test/target.jpg', bbox_inches='tight')
# plt.show()
plt.close(fig)

img2 = plt.imread('./test/rightimg.jpg')
fig = plt.figure()
plt.axis('off')
plt.imshow(img2)
plt.plot(match0, j, 'b.', linewidth=0.1)
plt.plot(match1, j, 'r.', linewidth=0.1)
plt.plot(match4, j, 'y.', linewidth=0.1)
plt.plot(match2, j, 'g.', linewidth=0.1)
plt.plot(match3, j, 'c.', linewidth=0.1)
fig.savefig('./test/source.jpg', bbox_inches='tight')
# plt.show()
plt.close(fig)

x1 = plt.imread('./test/1.jpeg')
# x1 = np.array(x1)
x1 = imresize(x1, 2)
plt.imshow(x1)
plt.show()

