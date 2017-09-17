import matplotlib.pyplot as plt 
import os
import numpy as np 
from scipy.optimize import curve_fit

from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline

# with open("./statis.txt") as f:
# 	data = f.read().splitlines()
# 	data = [x.split(" ") for x in data]

# print(data)

# data = np.array(data, dtype="float")
# # plt.plot(data[:,0], data[:,2])
# # plt.show()

def sigmoid(x, x0, k, a, b, c):
	y = a / (1 + np.exp(-k*(x-x0))) + c
	return y


def get_params(func, filename):
	with open(filename) as f:
		data = f.read().splitlines()
		data = [x.split(" ") for x in data]

	data = np.array(data, dtype="float")

	xdata = data[:, 0]
	ydata = data[:, 2]

	popt, _ =  curve_fit(sigmoid, xdata, ydata)

	return popt, xdata, ydata


# xdata = data[:, 0]
# ydata = data[:, 2]
# print(xdata)

# popt, pcov = curve_fit(sigmoid, xdata, ydata)
# print(popt)
# print(pcov)
# popt, xdata, ydata = get_params(sigmoid, "./statis.txt")

# x = np.linspace(1, 5, 50)
# y = sigmoid(x, *popt)

# plt.plot(xdata, (ydata-0.5)*2, 'o', label="data")
# plt.plot(x, 2*(y-0.5), label='fit')
# # plt.ylim
# plt.show()

# with open("./statis_dense.txt") as f:
# 	data = f.read().splitlines()
# 	data = [x.split(" ") for x in data]

# data = np.array(data, dtype="float")
# # print(data)
# # data = np.fliplr(data)
# # print(data)
# xdata = data[:, 0]
# ydata = data[:, 2]
# xdata = xdata[::-1]
# ydata = ydata[::-1]
# print(xdata, ydata)

# # spl = UnivariateSpline(xdata, ydata, s=1)
# spl = InterpolatedUnivariateSpline(xdata, ydata)
# # spl.set_smoothing_factor(1)
# xs = np.linspace(1, 5, 100)
# print(spl([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]))
# # plt.plot(xs, spl(xs), 'g', lw=3, label='fitted')
# plt.plot(xdata, ydata, 'o', label='data')
# # plt.xticks([2.0 + i*0.1 for i in range(9)])
# plt.legend()
# # plt.savefig("./statis_dense.jpg")
# plt.show()

with open("./statis.txt") as f:
	data = f.read().splitlines()
	data = [x.split(" ") for x in data]

data = np.array(data, dtype="float")
# print(data)
# data = np.fliplr(data)
# print(data)
xdata = data[:, 0]
ydata = data[:, 2]
xdata = xdata[::-1]
ydata = ydata[::-1]



spl = InterpolatedUnivariateSpline(xdata, ydata)
# spl.set_smoothing_factor(1)
xs = np.linspace(1, 5, 100)
print(2*(spl([2.0, 2.1, 2.2, 2.26, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0])-0.5))
plt.plot(xs, 2*(spl(xs)-0.5), 'g', lw=3, label='fitted data')
# plt.plot(xdata, ydata, 'o', label='parse data')
# plt.plot(xdata, ydata, 'o', label='data')

# with open("./statis_dense3.txt") as f:
# 	data = f.read().splitlines()
# 	data = [x.split(" ") for x in data]

# data = np.array(data, dtype="float")
# # print(data)
# # data = np.fliplr(data)
# # print(data)
# xdata = data[:, 0]
# ydata = data[:, 2]
# xdata = xdata[::-1]
# ydata = ydata[::-1]

# plt.plot(xdata, ydata, 'o', label='dense data')

# # spl = UnivariateSpline(xdata, ydata, s=1)

# # plt.xticks([2.0 + i*0.1 for i in range(9)])
# plt.legend()
# plt.savefig("./statis_dense3.jpg")
plt.show()