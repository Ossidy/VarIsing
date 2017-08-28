import torch
import torch.nn as nn

class Discriminator(nn.Module):
	def __init__(self, imsize, nz, nc, ndf, n_extra_layers=0):
		super(Discriminator, self).__init__()
		assert imsize % 16 == 0, "size has to be multiple of 16"

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
		main.add_module('final.{}-{}.conv'.format(cndf, 1),
						nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))

		self.main = main

	def forward(self, input):

		output = self.main(input)

		# output = output.mean(0)
		return output.view(input.size()[0])

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
						nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
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

	def forward(self, input):
		output = self.main(input)
		return output