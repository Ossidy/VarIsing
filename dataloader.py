import numpy as np 
import numpy.random as rd 
import matplotlib.pyplot as plt 
from matplotlib import colors
import random
from scipy.optimize import curve_fit

import json
import glob
import torch


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

def load_blobs(filename, is_json=False):
	print("Loading data from file: " + filename)
	with open(filename) as f:
		data = f.read().splitlines()
	data = [x.split(" ") for x in data]
	# print(data)
	print("Creating data blobs from file: " + filename)
	blobs = []
	for item in data:
		blob = {}
		blob["T"] = float(item[0])
		blob["mag"] = float(item[1])
		blob["mag_abs"] = float(item[2])
		blob["mag_sqr"] = float(item[3])
		blob["eng"] = float(item[4])
		# blob["lattice"] = [(float(i) + 0.0) / 1.0 for i in item[5:]]
		blob["lattice"] = np.array(item[5:], dtype="float")

		blob["lattice"] = resize_lattice(blob["lattice"])
		# print(blob["lattice"])

		blobs.append(blob)

	if is_json == True:
		print("Dumping to json file...")
		with open(filename[:-3]+"txt", 'w') as outfile:
			json.dump(blobs, outfile)
			# np.save(outf)

	print("Finished!!!")
	
	return blobs

def resize_lattice(lattice_array):
	# resize and rescale lattice value
	N = len(lattice_array)
	# if N != length * width:
	# 	raise("Dimension not compatible!!!")
	L = int(np.sqrt(N))
	# lattice = [(i+1)/2 for i in lattice_array]
	lattice = np.array(lattice_array)
	lattice = np.resize(lattice, (L, L))
	# print(lattice)
	return lattice.tolist()

def plot_lattice(lattice):
	cmap = colors.ListedColormap(['white', 'black'])
	bounds = [-1, 0., 1]
	norm = colors.BoundaryNorm(bounds, cmap.N)
	# print(lattice)

	fig, ax = plt.subplots()

	# plt.clf()
	
	ax.imshow(lattice, cmap=cmap, norm=norm)

	# ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
	ax.grid(True)
	ax.set_xticks(np.array([]))
	ax.set_yticks(np.array([]))

	plt.show()
	# plt.pause(1)

def dataloader(batchsize=16, datafile="./Ltest", fileformat=".dat", is_json=False, rep_time=1000, flip=True, rotate=True, reverse=True, dummy_dim=True):
	# print(data)
	if dummy_dim:
		popt = get_params(sigmoid, "./statis.txt")


	data_names = []
	while True:
		# random choose a data file 
		if data_names == []:
			data_names = glob.glob(datafile + "/*" + fileformat)
			assert data_names != [], "File not loaded, please check data file again!"

		# data = rd.choice(data_names)
		# data_names.remove(data) 
		blobs = []
		for data in data_names:
		# load the correspoding data file
			if is_json == True:
				# print(data)
				with open(data) as data_file:
					blobs.extend(json.load(data_file))
			else:
				blobs.extend(load_blobs(data))

			# take the length of blobs and do yield
		num_blobs = len(blobs)
		sample_time = num_blobs * rep_time // batchsize
		print("number of blobs: {}".format(num_blobs))

		print("shuffling blobs list for randomness...")
		random.shuffle(blobs)
		print("shuffling done!")
		# print(type(blobs))
		for i in range(sample_time):
			# print(i, num_blobs)
			inds = rd.choice(num_blobs, batchsize)

			output = []
			T_label = []
			for j in inds:
				lattice = blobs[j]["lattice"]
				# flip 
				if flip == True:
					rd_num = rd.uniform(0 ,1)
					if rd_num < 0.25:
						lattice = np.fliplr(lattice)
					elif rd_num >= 0.25 and rd_num < 0.5:
						lattice = np.flipud(lattice)
					else:
						lattice = np.array(lattice)
				# rotate
				if rotate == True:
					rot_num = rd.choice(4, 1)
					lattice = np.rot90(lattice, k=rot_num)
				# reverse 
				if reverse == True:
					rd_num = rd.uniform(0, 1)
					if rd_num < 0.5:
						lattice *= -1

				if dummy_dim:
					# add dummy dimension according to the abs_mag
					dummy_channel = np.zeros(shape=lattice.shape)
					dummy_channel.fill(2 * (sigmoid(blobs[j]["T"], *popt) - 0.5))
					# print(dummy_channel.shape)
					lattice_channel = np.expand_dims(lattice, axis=0)
					dummy_channel = np.expand_dims(dummy_channel, axis=0)
					composed = np.vstack((lattice_channel, dummy_channel))
				else:
					composed = np.expand_dims(lattice, axis=0)

				output.append(composed)
				T_label.append(blobs[j]["T"])

			# print(output)
			yield torch.FloatTensor(output), torch.FloatTensor(T_label)

if __name__ == "__main__":
	x = dataloader(datafile="./L32", dummy_dim=True)
	for i in range(10):
		# print(i)
		lattice, T = next(x)
		print(lattice)