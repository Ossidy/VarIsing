import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Discriminator(nn.Module):
	def __init__(self, imsize, nz, nc, ndf, n_extra_layers=0):
		super(Discriminator, self).__init__()
		assert imsize % 16 == 0, "size has to be multiple of 16"
		self.nc = nc

		main = nn.Sequential()
		main.add_module('initial.conv.{}-{}'.format(nc, ndf),
						nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
		main.add_module('initial.relu.{}'.format(ndf),
						nn.LeakyReLU(0.2, inplace=True))

		csize, cndf = imsize / 2, ndf
		# Extra layers
		for t in range(n_extra_layers):
			main.add_module('extra-layers-{}.{}.conv'.format(t, cndf),
							nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
			# main.add_module('extra-layers-{}.{}.batchnorm'.format(t, cndf),
			# 				nn.BatchNorm2d(cndf))
			main.add_module('extra-layers-{}.{}.relu'.format(t, cndf),
							nn.LeakyRelu(0.2, inplace=True))

		while csize > 4:
			in_feat = cndf
			out_feat = cndf * 2
			main.add_module('pyramid.{}-{}.conv'.format(in_feat, out_feat),
							nn.Conv2d(in_feat, out_feat,4, 2, 1, bias=False))
			# main.add_module('pyramid.{}.batchnorm'.format(out_feat),
			# 				nn.BatchNorm2d(out_feat))
			main.add_module('pyramid.{}.relu'.format(out_feat),
							nn.LeakyReLU(0.2, inplace=True))
			cndf = cndf * 2
			csize = csize / 2

		# state size. K x 4 x 4
		

		self.features = main 
		Dis_value = nn.Sequential()
		Dis_value.add_module('discrim-final-layer',
							  nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
		self.Dis_value = Dis_value

		Reg_value = nn.Sequential()
		Reg_value.add_module('Q-dense-layer-1',
							 nn.Conv2d(cndf, 500, 4, 1, 0, bias=False))
		Reg_value.add_module('flatten-layer',
							 Flatten())
		# Reg_value.add_module('Q-dense-layer-extra',
		# 					 nn.LeakyReLU(negative_slope=0.2))
		# Reg_value.add_module('Drop1',
		# 					 nn.Dropout())
		Reg_value.add_module('ReLu',
							 nn.LeakyReLU(negative_slope=0.01, inplace=False))
		Reg_value.add_module('Linear',
							 nn.Linear(500, 100, bias=True))
		# Reg_value.add_module('Drop2',
		# 					 nn.Dropout())

		Reg_value.add_module('ReLu-1',
							 nn.LeakyReLU(negative_slope=0.01, inplace=False))
		Reg_value.add_module('Q-dense-layer-2',
							 nn.Linear(100, 50, bias=True))
		Reg_value.add_module('ReLu-2',
							 nn.LeakyReLU(negative_slope=0.01, inplace=False))
		Reg_value.add_module('Q-dense-layer-3',
							 nn.Linear(50, 1, bias=True))
		self.Reg_value = Reg_value


		# main.add_module('final.{}-{}.conv'.format(cndf, 1),
		# 				nn.Conv2d(cndf, 2, 4, 1, 0, bias=False))

	def forward(self, input):

		output = self.features(input)
		Dis_val = self.Dis_value(output)
		Reg_val = self.Reg_value(output)
		# print(Reg_val)
		# output = output.mean(0)
		return Dis_val.view(input.size()[0]), Reg_val.view(input.size()[0])

class Generator(nn.Module):
	def __init__(self, imsize, nz, nc, ngf, n_extra_layers=0):
		super(Generator, self).__init__()
		assert imsize % 16 == 0, "image size has to be a multiple of 16"

		cngf, tisize = ngf//2, 4
		while tisize != imsize:
			cngf = cngf * 2
			tisize = tisize * 2

		main = nn.Sequential()
		# # input is Z, going into a conv
		main.add_module('initial.{}-{}.convt'.format(nz, cngf),
						nn.ConvTranspose2d(nz+1, cngf, 4, 1, 0, bias=False))
		# main.add_module('initial.{}.batchnorm'.format(cngf),
		# 				nn.BatchNorm2d(cngf))
		main.add_module('initial.{}.relu'.format(cngf),
						nn.ReLU(True))

		csize, cndf = 4, cngf
		while csize < imsize // 2:
			main.add_module('pyramid.{}-{}.convt'.format(cngf, cngf//2),
							nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
			# main.add_module('pyramid.{}.batchnorm'.format(cngf//2),
			# 				nn.BatchNorm2d(cngf//2))
			main.add_module('pyramid.{}.relu'.format(cngf//2),
							nn.ReLU(True))
			cngf = cngf // 2
			csize = csize * 2

		# Extra layers
		for t in range(n_extra_layers):
			main.add_module('extra-layers-{}.{}.conv'.format(t, cngf),
							nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
			# main.add_module('extra-layers-{}.batchnorm'.format(t, cngf),
			# 				nn.BatchNorm2d(cngf))
			main.add_module('extra-layers-{}.{}.relu'.format(t, cngf),
							nn.ReLU(True))

		main.add_module('final.{}-{}.convt'.format(cngf, nc),
						nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
		main.add_module('final.{}.tanh'.format(nc),
						nn.Tanh())

		self.main = main

	def forward(self, input, T):
		# print(input)
		# print(T)
		cat_input = torch.cat((input, T), dim=1)

		# print(cat_input)
		output = self.main(cat_input)
		return output


class InforQ(nn.Module):
	def __init__(self, imsize, nc, ndf, n_extra_layers=0):
		super(InforQ, self).__init__()
		assert imsize % 16 == 0, "size has to be multiple of 16"
		self.nc = nc

		main = nn.Sequential()
		main.add_module('initial.conv.{}-{}'.format(nc, ndf),
						nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
		main.add_module('initial.relu.{}'.format(ndf),
						nn.LeakyReLU(0.2, inplace=True))

		csize, cndf = imsize / 2, ndf
		# Extra layers
		for t in range(n_extra_layers):
			main.add_module('extra-layers-{}.{}.conv'.format(t, cndf),
							nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
			# main.add_module('extra-layers-{}.{}.batchnorm'.format(t, cndf),
			# 				nn.BatchNorm2d(cndf))
			main.add_module('extra-layers-{}.{}.relu'.format(t, cndf),
							nn.LeakyRelu(0.2, inplace=True))

		while csize > 4:
			in_feat = cndf
			out_feat = cndf * 2
			main.add_module('pyramid.{}-{}.conv'.format(in_feat, out_feat),
							nn.Conv2d(in_feat, out_feat,4, 2, 1, bias=False))
			# main.add_module('pyramid.{}.batchnorm'.format(out_feat),
			# 				nn.BatchNorm2d(out_feat))
			main.add_module('pyramid.{}.relu'.format(out_feat),
							nn.LeakyReLU(0.2, inplace=True))
			cndf = cndf * 2
			csize = csize / 2

		# state size. K x 4 x 4
		self.features = main
		
		Reg_value = nn.Sequential()
		Reg_value.add_module('Q-dense-layer-1',
							 nn.Conv2d(cndf, 500, 4, 1, 0, bias=False))
		Reg_value.add_module('flatten-layer',
							 Flatten())
		# Reg_value.add_module('Q-dense-layer-extra',
		# 					 nn.LeakyReLU(negative_slope=0.2))
		# Reg_value.add_module('Drop1',
		# 					 nn.Dropout())
		Reg_value.add_module('ReLu',
							 nn.LeakyReLU(negative_slope=0.01, inplace=False))
		Reg_value.add_module('Linear',
							 nn.Linear(500, 100, bias=True))
		# Reg_value.add_module('Drop2',
		# 					 nn.Dropout())

		Reg_value.add_module('ReLu-1',
							 nn.LeakyReLU(negative_slope=0.01, inplace=False))
		Reg_value.add_module('Q-dense-layer-2',
							 nn.Linear(100, 50, bias=True))
		Reg_value.add_module('ReLu-2',
							 nn.LeakyReLU(negative_slope=0.01, inplace=False))
		Reg_value.add_module('Q-dense-layer-3',
							 nn.Linear(50, 1, bias=True))
		self.Reg_value = Reg_value


		# main.add_module('final.{}-{}.conv'.format(cndf, 1),
		# 				nn.Conv2d(cndf, 2, 4, 1, 0, bias=False))

	def forward(self, input):

		output = self.features(input)
		Reg_val = self.Reg_value(output)
		# print(Reg_val)
		# output = output.mean(0)
		return Reg_val.view(input.size()[0])