import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()  #定义新的三维坐标轴
ax3 = plt.axes(projection='3d')

#定义三维数据
xx = np.arange(-5,5,0.1)
yy = np.arange(-5,5,0.1)
X, Y = np.meshgrid(xx, yy)

Z = 20 + X ** 2 + Y ** 2 - 10 * (np.cos(0.5 * np.pi * X) + np.cos(0.5 * np.pi * Y))


#作图
ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1,cmap='rainbow')
# ax3.contour(X,Y,Z,offset=-2, cmap = 'rainbow')#绘制等高线
plt.show()