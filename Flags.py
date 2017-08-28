
class Flags():
	def __init__(self):

		self.niters = 100000
		self.Diters = 5
		self.D_learning_rate = 1e-4
		self.G_learning_rate = 1e-4
		self.batch_size = 32
		self.weight_penalty = 10

		self.dim_noise = 100
		self.dim_label = 2

		self.image_size = 32
		self.ngf = 64 # number of generator features
		self.ndf = 64 # number of discriminator features
		self.nc = 2 # image channel
		
		self.n_extra_layers = 0 # extra_layers in netG and netD

		self.optim = 'Adam'

		self.save_path = './saved_models/'
		self.load_path = './saved_models/'
		self.offset = 300
		self.save_time = 500

		self.sampler = "Inf_well"

	def show_info(self):
		if self.load_path == None:
			print("******************************************************************")
			print("No pretrained model designated, new model will be set...")
		else:
			print("Model will be loaded from {}".format(self.load_path))
			print("******************************************************************")

		if self.save_path == None:
			print(" ")
			print("Attention!!! Model will not be saved!!!")
			print("******************************************************************")
		else:
			print(" ")
			print("Model will be saved into {} for every {} iterations".format(self.save_path, self.save_time))
			print("******************************************************************")

		print(' ')
		print("network information:")
		print("iteration times: \t {}".format(self.niters))
		print("learning rate for D: \t {}".format(self.D_learning_rate))
		print("learning rate for G: \t {}".format(self.G_learning_rate))
		print("batch size: \t \t {}".format(self.batch_size))
		print("iterations for D: \t {}".format(self.Diters))
		print("weight penalty: \t {}".format(self.weight_penalty))
		print("optimizatio method: \t {}".format(self.optim))
		print("dimension of noise: \t {}".format(self.dim_noise))
		print("dimension of label: \t {}".format(self.dim_label))
		print("image size: \t \t {}".format(self.image_size))
		print("num of feature in G: \t {}".format(self.ngf))
		print("num of feature in D: \t {}".format(self.ndf))
		print("extra layers: \t \t {}".format(self.n_extra_layers))

		# if self.image_prefix == None:
		# 	print("image will be saved: \t {}".format("False"))
		# else:
		# 	print("image will be saved: \t {}".format("True"))

		print(' ')
		print("Sampler information:")
		print("Sampler type: \t \t {}".format(self.sampler))
		print("Sampler parameters: \t {}".format("Undefined"))

		print(' ')

		print("******************************************************************")


if __name__ == "__main__":
	a = Flags()
	a.show_info()