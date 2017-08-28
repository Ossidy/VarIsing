
import torch
import numpy as np 
import os

import model as dcgan


def save_model(netD, netG, losses, savepath='./', name=0):
	# save net models
	if not os.path.exists(savepath):
		os.makedirs(savepath)

	path_to_save = savepath + str(name) + '/'
	if os.path.exists(path_to_save):
		print("File path {} has already been created, model will not be save unless delete the directory".format(savepath + name + '/'))
		return 0
	
	os.makedirs(path_to_save)
	torch.save(netD.state_dict(), path_to_save + 'netD.pth')
	torch.save(netG.state_dict(), path_to_save + 'netG.pth')
	np.save(path_to_save + "losses", losses)


def load_model(imsize, nz, nc, ndf, ngf, n_extra_layers, savepath):
	# load net models
	print(savepath)
	assert (os.path.exists(savepath)==True)
	netD_name = 'netD.pth'
	netG_name = 'netG.pth'

	print(netD_name, netG_name)
	netD = dcgan.Discriminator(imsize = imsize, nz=nz, nc=nc, ndf=ndf, n_extra_layers=n_extra_layers)
	netG = dcgan.Generator(imsize = imsize, nz=nz, nc=nc, ngf=ngf, n_extra_layers=n_extra_layers)

	netD.load_state_dict(torch.load(savepath + netD_name))
	netG.load_state_dict(torch.load(savepath + netG_name))

	losses = np.load(savepath + "losses.npy")
	return netD, netG, list(losses)

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