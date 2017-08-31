import argparse
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

from torch.autograd import Variable
import torch.autograd as autograd

import os
import numpy as np

import model as dcgan
import dataloader
import ImageGenerator
import Flags

from utils import save_model, load_model

cudnn.benchmark = True


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.01)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

def get_observables(lattice):
	# in terms of numpy array
	shape = lattice.shape
	mag = 0
	mag_int = 0
	mag_abs = 0
	mag_sqr = 0
	eng = 0
	for i in range(shape[0]):
		for j in range(shape[1]):
			# print(np.floor(lattice[i][j]) * 2. + 1. )
			mag += lattice[i][j]
			if lattice[i][j] <= 0:
				mag_int -= 1.
			else:
				mag_int += 1.
	mag = mag / float(lattice.size)
	mag_int = mag_int / float(lattice.size)
	return mag, mag_int

def get_rd_tensor(batch_size, image_size, nc):
	a = np.random.uniform(0, 1, size=(batch_size, 1))
	b = np.zeros((batch_size, nc, image_size, image_size))
	for i in range(batch_size):
		b[i,:,:,:] = a[i]
	return torch.FloatTensor(b).cuda()

def get_grad_penalty(netD, real_data, fake_data, batch_size, image_size, nc):
	alpha = get_rd_tensor(batch_size, image_size, nc)
	# print(alpha)
	alpha = alpha.cuda()
	inter_data = alpha * real_data + ((1 - alpha) * fake_data)

	inter_data = Variable(inter_data, requires_grad=True)
	inter_D_out, _ = netD(inter_data)

	inter_D_grad = autograd.grad(outputs=inter_D_out, inputs=inter_data, grad_outputs=torch.ones(inter_D_out.size()).cuda(),
								 create_graph=True, retain_graph=True, only_inputs=True)[0]


	grad_norm = torch.sum((inter_D_grad**2), dim=3)
	grad_norm = torch.sum(grad_norm, dim=2)
	grad_norm = torch.sum(grad_norm, dim=1)
	grad_norm = torch.sqrt(grad_norm)

	loss_penalty = ((grad_norm - 1)**2).mean()

	return loss_penalty


def train(niters=20000, batch_size=16, lr_D=0.00005, lr_G=0.00005, Diter=5, weight_penalty=10,
		  dim_noise=100, image_size=32, ndf=64, ngf=64, nc=2, n_extra_layers=0,
		  save_model_path=None, load_model_path=None, save_image_path=None, offset=0):

	clip_fn = lambda x:x.clamp(max=0)

	if load_model_path == None:
		netD = dcgan.Discriminator(image_size, dim_noise, nc, ndf, n_extra_layers)
		netG = dcgan.Generator(image_size, dim_noise, nc, ngf, n_extra_layers)
		losses = []
		netD.apply(weights_init)
	else:
		netD, netG, losses= load_model(image_size, dim_noise, nc, ndf, ngf, n_extra_layers, load_model_path)
	# netD.apply(weights_init)

	real_data = Variable(torch.FloatTensor(batch_size, nc, image_size, image_size), requires_grad=False)
	gen_input = Variable(torch.FloatTensor(batch_size, dim_noise, 1, 1), requires_grad=False)
	T_input = Variable(torch.FloatTensor(batch_size, 1, 1, 1), requires_grad=False)
	# fake_data = Variable(torch.FloatTensor(batch_size, 3)).cuda()
	
	# mone = -1 * one

	netD.cuda()
	netG.cuda()

	########################################
	# test netD.parameters()
	# print(netD.Reg_value.parameters())
	# print(netD.Dis_value.parameters())
	print(netD.features)
	# print(netD.parameters())
	# reg = 0
	# for p in netD.Reg_value.parameters():
	# 	reg += 1
	# dis = 0
	# for p in netD.Dis_value.parameters():
	# 	dis += 1
	# fea = 0
	# for p in netD.features.parameters():
	# 	fea += 1
	# tot = 0
	# for p in netD.parameters():
	# 	tot += 1 

	# print(reg, dis, fea, tot)
	########################################

	# print(netD)
	# print(netG)
	real_data = real_data.cuda()
	# one, mone = one.cuda(), mone.cuda()
	gen_input = gen_input.cuda()
	T_input = T_input.cuda()

	optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(0.5, 0.999))
	optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(0.5, 0.999))

	gen_iterations = 0

	image_generator = ImageGenerator.ImageGenerator(save_path=save_image_path)
	data_iter = dataloader.dataloader(batchsize=batch_size, fileformat='.dat', datafile="./L32", is_json=False, dummy_dim=True) # function should be defined as iterator for images

	for epoch in range(niters):
		TEMP = 0
		# update D network
		# print(netD.Reg_value)
		for p in netD.features.parameters():
			p.requires_grad = True
		for p in netD.Dis_value.parameters():
			p.requires_grad = True
		# for p in netD.Reg_value.parameters():
		# 	p.requires_grad =False

		if gen_iterations < 100 or gen_iterations % 500 == 0:
			Diters = 100
		else:
			Diters = Diter

		for i in range(Diters): # need change !!!!!!!!!!!!!!!!!!!

			data, T = next(data_iter)
			data = data.cuda()
			T = T.cuda()
			T.view(-1, 1, 1, 1)
			# data = 0
			netD.zero_grad()
			# print(next(netD.parameters()).data)

			real_data.data.copy_(data)

			errD_real_vec, pred_T_vec = netD(real_data)
			errD_real = errD_real_vec.mean(0)
			# errD_real.backward(one)

			gen_input.data.uniform_(-1,1)
			T_input.data.copy_(T)
			# fake = Variable(netG(noisev).data)
			fake_data = netG(gen_input, T_input)

			errD_fake_vec, _ = netD(fake_data)
			# print(errD_fake_vec)
			errD_fake = errD_fake_vec.mean(0)
			# print(errD_fake)

			errD = errD_fake - errD_real #+ loss_penalty
			errD.backward()

			loss_penalty = weight_penalty * get_grad_penalty(netD, real_data.data, fake_data.data, batch_size=batch_size, image_size=image_size, nc=nc)
			loss_penalty.backward()

			optimizerD.step()
			TEMP = [errD_real, errD_fake, loss_penalty]

		# update Q network
		# fix D network
		for p in netD.features.parameters():
			p.requires_grad = False
		for p in netD.Dis_value.parameters():
			p.requires_grad = False
		for p in netD.Reg_value.parameters():
			p.requires_grad = True

		if gen_iterations < 100 or gen_iterations % 500 == 0:
			Qiters = 1000
		else:
			Qiters = Diter



		for i in range(Qiters):
			data, T = next(data_iter)
			data = data.cuda()
			netD.zero_grad()
			T = T.cuda()
			T = Variable(T, requires_grad=False)
			# print(T)
			real_data.data.copy_(data)
			_, pred_T_vec = netD(real_data)
			# print((T - pred_T_vec)**2)
			loss_T = 100 * ((T - pred_T_vec)**2).mean(0)
			# print(loss_T)
			loss_T.backward()

			optimizerD.step()

			TEMP.append(loss_T)

			if (i+1) % 500 == 0:
				print(pred_T_vec.data[0], T.data[0], loss_T.data[0])

		# if gen_iterations <= 5:
		# 	while loss_T.data[0] > 1:
		# 		data, T = next(data_iter)
		# 		data = data.cuda()
		# 		netD.zero_grad()
		# 		T = T.cuda()
		# 		T = Variable(T, requires_grad=False)
		# 		# print(T)
		# 		real_data.data.copy_(data)
		# 		_, pred_T_vec = netD(real_data)
		# 		# print((T - pred_T_vec)**2)
		# 		loss_T = 100 * ((T - pred_T_vec)**2).mean(0)
		# 		# print(loss_T)
		# 		loss_T.backward()

		# 		optimizerD.step()

		# 		TEMP.append(loss_T)
		# 		print(pred_T_vec.data[0], T.data[0], loss_T.data[0])

		# for i in range(Qiters):

			
		# update G network
		for p in netD.Reg_value.parameters():
			p.requires_grad = False

		netG.zero_grad()
		gen_input.data.uniform_(-1,1)
		T_input.data.uniform_(2., 3)

		fake_data = netG(gen_input, T_input)

		errG_vec, pred_T_vec = netD(fake_data)
		errG = -errG_vec.mean(0)
		# print(pred_T_vec)
		# print(T_input)
		# print(loss_T)
		# print((T_input.view(-1))**2)
		# print(pred_T_vec)
		loss_T = 100 * ((T_input.view(-1) - pred_T_vec)**2).mean(0)
		# print(loss_T)
		# print(errG)
		errG += loss_T
		errG.backward()

		optimizerG.step()
		gen_iterations += 1

		if gen_iterations % 50 == 0:
			# print(np.array(fake.cpu().data.tolist()[0][0]))
			# loss with sequence: 
			# 1. W distance of two distribution -- errD
			# 2. value of real from critics -- errD_real
			# 3. value of fake from critics -- errD_fake
			# 4. value of generated data from critics -- errG
			# 5. value of weight penalty
			# print([errD.cpu().data.numpy()[0], TEMP[0].cpu().data[0], TEMP[1].cpu().data[0], errG.cpu().data[0], TEMP[2].cpu().data[0]])
			losses.append([errD.cpu().data.numpy()[0], TEMP[0].cpu().data[0], TEMP[1].cpu().data[0], errG.cpu().data[0], TEMP[2].cpu().data[0], TEMP[3].cpu().data[0]])
			print("iter: {}, errD: {}, errD_real: {}, errD_fake: {}, errG: {}, err_T: {}".format(gen_iterations, errD.cpu().data[0], TEMP[0].cpu().data[0], TEMP[1].cpu().data[0], errG.cpu().data[0], TEMP[3].cpu().data[0]))
			print(T_input.cpu().data[0].numpy(), np.mean(fake_data.cpu().data.numpy()[0,1,:,:]), get_observables(np.array(fake_data.cpu().data.tolist()[0][0])))
			# print(gen_iterations, errD.cpu().data[0], TEMP.cpu().data[0], get_observables(np.array(fake_data.cpu().data.tolist()[0][0])))
			# print(fake.cpu().data.tolist())
			image_generator(fake_data.cpu().data.tolist()[0][0], losses)

		if gen_iterations % 500 == 0:
			# save model
			save_model(netD, netG, losses, savepath=save_model_path, name=str(gen_iterations + offset))

	return losses

if __name__ == "__main__":

	flags = Flags.Flags()

	flags.show_info()

	train(niters=flags.niters, batch_size=flags.batch_size, lr_D=flags.D_learning_rate, lr_G=flags.G_learning_rate, 
		  Diter=flags.Diters, weight_penalty=flags.weight_penalty, dim_noise=flags.dim_noise,
		  image_size=flags.image_size, ndf=flags.ndf, ngf=flags.ngf, nc=flags.nc, n_extra_layers=flags.n_extra_layers,
		  save_model_path="./saved_models/", save_image_path="./saved_images/", load_model_path="./saved_models_stage1/94000/")
