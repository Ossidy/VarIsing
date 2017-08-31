import torch
from torch.autograd import Variable

from utils import load_model
import ImageGenerator
import Flags

import numpy as np 
import numpy.random as rd 
import matplotlib.pyplot as plt

import sys

def test(image_size, nz, nc, ndf, ngf, n_extra_layers, num_sample=2, load_model_path='./saved_models/500/'):

	netD, netG, losses = load_model(image_size, nz, nc, ndf, ngf, n_extra_layers, load_model_path)

	# print(list(losses))

	gen_input = Variable(torch.FloatTensor(num_sample, nz, 1, 1), requires_grad=False)
	gen_input.data.uniform_(-1,1)

	T_input = Variable(torch.FloatTensor(num_sample, 1, 1, 1), requires_grad=False)
	# gen_input.data = torch.FloatTensor(torch.ones(num_sample, nz, 1,1))
	T = torch.FloatTensor([2.5])
	T.view(-1, 1, 1, 1)
	T_input.data.copy_(T)
	print(T)
	fake_data = netG(gen_input, T_input)
	
	return fake_data.data

def generator(T, image_size, nz, nc, ndf, ngf, n_extra_layers, num_sample=1, load_model_path='./saved_models/100000/'):
	netD, netG, losses = load_model(image_size, nz, nc, ndf, ngf, n_extra_layers, load_model_path)

	gen_input = Variable(torch.FloatTensor(num_sample, nz, 1, 1), requires_grad=False)
	T_input = Variable(torch.FloatTensor(num_sample, 1, 1, 1))
	T = torch.FloatTensor([T])
	T.view(-1, 1, 1, 1)
	T_input.data.copy_(T)

	# print(gen_input)
	while True:
		gen_input.data.uniform_(-1,1)
		# print(gen_input.data)
		fake_data = netG(gen_input, T_input)
		for i in range(num_sample):
			yield torch.unsqueeze(fake_data[i].data, 0)

def get_observables(lattice):
	# in terms of numpy array
	shape = lattice.shape
	mag = 0
	mag_int = 0
	mag_abs = 0
	mag_sqr = 0
	eng = 0
	pos = 0
	for i in range(shape[0]):
		for j in range(shape[1]):
			# print(np.floor(lattice[i][j]) * 2. + 1. )
			mag += lattice[i][j]
			if lattice[i][j] <= 0:
				mag_int -= 1.
			else:
				mag_int += 1.
				pos += 1
			# mag += np.floor(lattice[i][j]) * 2. + 1. 
	# print(lattice)
	print(pos)
	print(lattice.size)
	mag = mag / float(lattice.size)
	mag_int = mag_int / float(lattice.size)
	return mag, mag_int

if __name__ == "__main__":

	flags = Flags.Flags()
	fake_data = test(image_size=flags.image_size, nz=flags.dim_noise, nc=flags.nc, ndf=flags.ndf, ngf=flags.ngf,
					 n_extra_layers=flags.n_extra_layers,
					 num_sample=16, load_model_path='./saved_models/100000/')
	# print(np.squeeze(fake_data.numpy()))
	mag, mag_int = get_observables(np.squeeze(fake_data[0,0].numpy()))
	print(mag, mag_int)


	ImGen = ImageGenerator.ImageGenerator(sustain=True, show_loss=False)
	mag_list = []

	# for i in range(16):
	print(np.mean(fake_data.numpy()[0,1,:,:]))
	ImGen(fake_data.numpy()[0,1,:,:])