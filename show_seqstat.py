import matplotlib.pyplot as plt 
import os
import numpy as np 
from scipy.optimize import curve_fit

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

	return popt

# xdata = data[:, 0]
# ydata = data[:, 2]
# print(xdata)

# popt, pcov = curve_fit(sigmoid, xdata, ydata)
# print(popt)
# print(pcov)
popt = get_params(sigmoid, "./statis.txt")

x = np.linspace(1, 5, 50)
y = sigmoid(x, *popt)

# plt.plot(xdata, ydata, 'o', label="data")
plt.plot(x, y, label='fit')
# plt.ylim
plt.show()