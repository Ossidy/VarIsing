import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab 
import torch

from test import generator
import dataloader
import Flags

class Measure():
	def __init__(self, data_iter=None ,num_bins=100):
		# if data_iter != None:
		self.data_iter = data_iter

		shape = next(self.data_iter).size()
		assert shape[0] == 1, "batch size of dataloader must be 1!"

		# shape[1] is channel and will not be used here
		self.length = shape[2]
		self.width = shape[3]

		# statistics of observables
		self.mag = []
		self.eng = []
		self.abs_mag = []

		# statistical parameters
		# self.is_abs_mag = False
		self.num_bins = num_bins

	def _latticeRegularization(self, lattice):
		lattice[lattice < 0] = -1.
		lattice[lattice > 0] = 1.
		return lattice

	def _getMagetization(self, lattice):
		mag = 0
		# print(lattice)
		for i in range(self.length):
			for j in range(self.width):
				mag += lattice[i,j]
		mag = mag / float(self.length * self.width)
		return mag

	def _getEnergy(self, lattice):
		eng = 0
		for i in range(self.length):
			for j in range(self.width):
				# print(i, (j-1)%self.width, type(i))
				eng += - lattice[i,j] * (lattice[(i+1)%self.length, j] +
										 lattice[i, (j+1)%self.width] +
										 lattice[(i-1)%self.length, j] +
										 lattice[i, (j-1)%self.width])
		eng = eng / (self.length * self.width * 2)
		# print(eng)
		return eng

	def statisObs(self, num_sample=100):
		# clear all sotored measures
		self.mag = []
		self.eng = []
		self.abs_mag = []
		for i in range(num_sample):
			# get lattice
			lattice = next(self.data_iter)
			lattice = np.squeeze(lattice.numpy()[0,0])
			lattice = self._latticeRegularization(lattice)

			# get observables:
			mag = self._getMagetization(lattice)
			self.mag.append(mag)
			self.abs_mag.append(abs(mag))
			self.eng.append(self._getEnergy(lattice))


	def histMagnetization(self, is_abs=True, save_name=None):
		
		fig, ax = plt.subplots()
		if is_abs == True:
			n, bins, patches = ax.hist(self.abs_mag, self.num_bins, normed=True)
		else:
			n, bins, patches = ax.hist(self.mag, self.num_bins, normed=True)
		ax.set_xlabel('Magnetization')
		ax.set_ylabel('Probability density')
		ax.set_title(r'Histogram of Magnetization')
		if save_name != None:
			plt.savefig(save_name)
		plt.show()


	def histEnergy(self, save_name=None):
		fig, ax = plt.subplots()
		n, bins, patches = ax.hist(self.eng, self.num_bins, normed=True)
		ax.set_xlabel('Energy')
		ax.set_ylabel('Probability density')
		ax.set_title(r'Histogram of Energy')
		fig.tight_layout()
		if save_name != None:
			plt.savefig(save_name)
		plt.show()


if __name__ == "__main__":
	data_iter = dataloader.dataloader(batchsize=1, datafile="./data", fileformat=".dat", is_json=False, rep_time=100, flip=False, rotate=False, reverse=False)
	# print(next(data_iter))
	flags = Flags.Flags()
	data_iter = generator(T = 3.0, image_size=flags.image_size, nz=flags.dim_noise, nc=flags.nc, ndf=flags.ndf, ngf=flags.ngf,
					 	  n_extra_layers=flags.n_extra_layers,
						  num_sample=32, load_model_path='./saved_models/100000/')

	mes = Measure(data_iter)
	mes.statisObs(10000)
	mes.histMagnetization(save_name="./test.jpg")
	# mes.histEnergy(save_name="./eng_GP_30000.jpg")
	# mes.histEnergy()
	# mes.histMagnetization(is_abs=True)