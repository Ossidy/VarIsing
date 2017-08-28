import matplotlib.pyplot as plt 
from matplotlib import colors
import matplotlib as mpl 

import numpy as np 
import torch 
from torch.autograd import Variable

import os

import dataloader

def get_observables(lattice):
	# in terms of numpy array
	shape = lattice.shape
	mag = 0
	mag_abs = 0
	mag_sqr = 0
	eng = 0
	# print(shape)
	for i in range(shape[0]):
		for j in range(shape[1]):
			# print(np.floor(lattice[i][j]) * 2. + 1. )
			# mag += np.floor(lattice[i][j]) * 2. + 1. 
			mag += lattice[i][j]
			# print(i,j, lattice[i][j])
			# print(np.floor(lattice[i][j]) * 2. + 1. )
	# print(lattice)
	# print(lattice.size)
	print(mag)
	mag = mag / float(lattice.size)
	return mag

class ImageGenerator:
	def __init__(self, prefix='frame', is_show=True, save_path=None, sustain=False, show_loss=True, regularization=False):
		self.prefix = prefix
		self.frame_index = 1
		self.save_path = save_path
		self.is_show = is_show
		self.sustain = sustain
		self.show_loss = show_loss
		self.regularization = regularization

		if self.show_loss:
			self.fig, (self.ax, self.ax2) = plt.subplots(1, 2)
		else:
			self.fig, self.ax = plt.subplots()

		

		if self.save_path != None:
			if not os.path.exists(self.save_path):
				os.makedirs(self.save_path)

	def __call__(self, lattice, losses=None):


		# print(lattice)

		cmap = mpl.cm.gray
		norm = colors.Normalize(vmin=-1, vmax=1)

		if self.regularization == True:
			cmap = colors.ListedColormap(['white', 'black'])
			bounds = [-1.1, 0., 1.1]
			norm = colors.BoundaryNorm(bounds, cmap.N)

	# fig, ax = plt.subplots()

		# plt.clf()
	
		self.ax.imshow(lattice, cmap=cmap, norm=norm)

	# ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
		self.ax.grid(True)
		self.ax.set_xticks(np.array([]))
		self.ax.set_yticks(np.array([]))

		if self.show_loss:
			assert losses != [], "losses is not passed!"
			plt.cla()
			losses = np.array(losses)
			# print(losses[0,0])
			self.ax2.plot(abs(losses[:,0]), label="W-distance")
			self.ax2.plot(abs(losses[:,0] + losses[:,4]), label="W-distance w/ penal")
			self.ax2.plot(abs(losses[:,1]), label="errD-real")
			self.ax2.plot(abs(losses[:,2]), label="errD-fake")
			self.ax2.plot(abs(losses[:,3]), label="errG")
			plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='x-small')
			# self.ax2.square()

		if self.save_path != None:
			plt.savefig(self.save_path + self.prefix+'{:05d}'.format(self.frame_index)+'.jpg')
		self.frame_index += 1

		if self.is_show == True:
			if self.sustain == True:
				# sustain doesn't matter if ImageGenerator is not destructed (called in functions repeatedly)
				plt.pause(100000)
			else:
				plt.pause(1)


if __name__=="__main__":
	data_iter = dataloader.dataloader(batchsize=1, fileformat='.dat', is_json=False)
	ImGen = ImageGenerator()
	# print(next(data_iter)[0])
	while True:
		lattice = next(data_iter)[0].tolist()[0]
		# print(next(data_iter)[0].tolist()[0])
		print(get_observables(np.array(lattice)))
		# print(next(data_iter)[0].tolist()[0])
		ImGen(lattice)
	# a = [[1, 1, 1, 1], [-1, -1, -1, -1], [1, 1, 1, 1], [-1, -1, -1, -1]]
	# b = [[-1, 1, 1, 1], [-1, -1, -1, -1], [1, 1, 1, 1], [-1, -1, -1, -1]]
	# ImGen(a)
	# ImGen(b)