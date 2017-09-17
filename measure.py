import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab 
import torch
import os

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
		self.eng_sqr = []
		self.abs_mag = []
		self.mag_sqr = []
		self.ave1 = []
		self.ave2 = []
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

	def _getAve(self, lattice):
		ave = 0
		for i in range(self.length):
			for j in range(self.width):
				ave += lattice[i, j]
		ave = ave / (self.length * self.width)
		# print(ave)
		return ave 


	def statisObs(self, num_sample=100):
		# clear all sotored measures
		self.mag = []
		self.eng = []
		self.abs_mag = []
		for i in range(num_sample):
			# get lattice
			lattice_ = next(self.data_iter)
			lattice = np.squeeze(lattice_.numpy()[0,0])
			lattice = self._latticeRegularization(lattice)
			lattice_fake = lattice_.numpy()[0,1]

			# get observables:
			mag = self._getMagetization(lattice)
			self.mag.append(mag)
			self.abs_mag.append(abs(mag))
			self.eng.append(self._getEnergy(lattice))
			# self.ave1.append(self._getAve(lattice))
			# self.ave2.append(self._getAve(lattice_fake))
			# print(self._getAve(lattice_fake))

	def statisMag(self, num_sample=100):
		self.mag = []
		self.abs_mag = []

		for i in range(num_sample):
			lattice_ = next(self.data_iter)
			lattice = np.squeeze(lattice_.numpy()[0,0])
			lattice = self._latticeRegularization(lattice)

			# get observables:
			mag = self._getMagetization(lattice)
			self.mag.append(mag)
			self.abs_mag.append(abs(mag))
			# self.mag_sqr.append(mag**2)

	def statisEng(self, num_sample=100):
		self.eng = []

		for i in range(num_sample):
			# get lattice
			lattice_ = next(self.data_iter)
			lattice = np.squeeze(lattice_.numpy()[0,0])
			lattice = self._latticeRegularization(lattice)
			# get observables:
			eng = self._getEnergy(lattice)
			self.eng.append(eng)
			# self.eng_sqr.append(eng**2)

	def statisAve(self, num_sample=100):
		self.ave1 = []
		self.ave2 = []

		for i in range(num_sample):
			# get lattice
			lattice_ = next(self.data_iter)
			lattice = np.squeeze(lattice_.numpy()[0,0])
			lattice = self._latticeRegularization(lattice)
			lattice_fake = lattice_.numpy()[0,1]

			# get observables:
			self.ave1.append(self._getAve(lattice))
			self.ave2.append(self._getAve(lattice_fake))


	def histMagnetization(self, is_abs=True, save_name=None):
		
		fig, ax = plt.subplots()

		if is_abs == True:
			weights = np.ones_like(self.abs_mag) / float(len(self.abs_mag))

			n, bins, patches = ax.hist(self.abs_mag, self.num_bins, weights=weights)
		else:
			weights = np.ones_like(self.mag) / float(len(self.mag))
			n, bins, patches = ax.hist(self.mag, self.num_bins, weights=weights)
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

	def histAve(self, layer=2):

		fig, ax = plt.subplots()
		if layer == 1:
			n, bins, patches = ax.hist(self.ave1, self.num_bins, normed=True)
		elif layer == 2:
			n, bins, patches = ax.hist(self.ave2, self.num_bins, normed=True)
		ax.set_xlabel('Energy')
		ax.set_ylabel('Probability density')
		ax.set_title(r'Histogram of Energy')
		fig.tight_layout()
		plt.show()

def writeNumpy(array, filepath):
	pass 

def outputResults(start=2.0, end=2.7, interval=0.1, num_sample=10000, trim=None, outpath="./Results/"):

	if not os.path.exists(outpath):
		os.makedirs(outpath)

	flags = Flags.Flags()
	T = start

	# do T = 2.26 first 
	# print(2.26)
	special_list = [2.26, 2.33, 2.37, 2.44]
	special_list = [2.3, 2.4, 2.5]
	for temper in special_list:
		print(temper)

		data_iter = generator(T = temper, trim = trim, image_size=flags.image_size, nz=flags.dim_noise, nc=flags.nc, ndf=flags.ndf, ngf=flags.ngf,
					 	  n_extra_layers=flags.n_extra_layers,
						  num_sample=1, load_model_path="./saved_models_penalty_mag_var3/100000/")
		mes = Measure(data_iter)
		mes.statisObs(num_sample)

		out = [mes.mag, mes.abs_mag, mes.eng]
		np.save(outpath+"{}".format(temper), out)

	# while T < end+1e-5:
	# 	print(T)
	# 	data_iter = generator(T = T, trim = trim, image_size=flags.image_size, nz=flags.dim_noise, nc=flags.nc, ndf=flags.ndf, ngf=flags.ngf,
	# 				 	  n_extra_layers=flags.n_extra_layers,
	# 					  num_sample=1, load_model_path="./saved_models_penalty_mag_var3/100000/")
	# 	mes = Measure(data_iter)
	# 	mes.statisObs(num_sample)
	# 	out = [mes.mag, mes.abs_mag, mes.eng]
	# 	np.save(outpath+"{}".format(T), out)
	# 	T += interval
	# 	T = round(T, 2)

def get_all_ave(filepath, is_plot=True):
	import glob
	# print(filepath + "/*" + "npy")
	len_filename = len(filepath)
	# print(len_filename)
	array_names = glob.glob(filepath + "/*" + "npy")
	statis = {}
	# print(array_names)
	for array in array_names:
		T = float(array[len_filename+1:-4])
		statis_array = np.load(array)
		statis[T] = np.array(statis_array)
	# print(statis)

	all_T = list(statis.keys())
	all_ave = []
	all_var = []
	for T in all_T:
		all_ave.append((statis[T].mean()))
		all_var.append((statis[T].var()))
	print(all_ave)
	print(all_T)
	print(all_var)
	if is_plot:
		from utils import get_spline_interp
		interp = get_spline_interp()

		xs = np.linspace(1, 5, 100)
		ys = interp(xs)
		plt.plot(xs, ys, "b")
		plt.plot(all_T, all_ave, "*g")
		plt.plot(all_T, all_var, "o")
		plt.savefig(filepath+"/compare.jpg")
		plt.show()

if __name__ == "__main__":
	# # data_iter = dataloader.dataloader(batchsize=1, datafile="./L32", fileformat="T280.dat", is_json=False, rep_time=100, flip=False, rotate=False, reverse=False)
	# # print(next(data_iter))
	# flags = Flags.Flags()
	# data_iter = generator(T = 2.40, trim=-0.01, image_size=flags.image_size, nz=flags.dim_noise, nc=flags.nc, ndf=flags.ndf, ngf=flags.ngf,
	# 				 	  n_extra_layers=flags.n_extra_layers,
	# 					  num_sample=1, load_model_path="./saved_models_penalty_mag_var3/100000/")

	# mes = Measure(data_iter)
	# mes.statisMag(10000)
	# mes.histMagnetization(save_name="./test2.4_ini.jpg")
	# # mes.histEnergy(save_name="./eng_GP_30000.jpg")
	# # mes.histEnergy()
	# # mes.histMagnetization(is_abs=True)	
	# # mes.histAve(layer=2)

	outputResults(outpath="./Results_additional/", trim=0.1, interval=0.01)
	# get_all_ave(filepath="./Results_trim_0.45")
	# get_all_ave(filepath="./Results")
	# abs_mag = np.load("./Results_trim_0.01/2.2.npy")
	# n, bins, patches = plt.hist(abs_mag, 50, normed=1)
	# print(abs_mag.mean())
	# plt.show()