import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR,SSIM
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from PIL import Image
import torch
import torchvision.transforms as transforms



if __name__ == '__main__':


	opt = TestOptions().parse()
	opt.nThreads = 1   # test code only supports nThreads = 1
	opt.batchSize = 1  # test code only supports batchSize = 1
	opt.serial_batches = True  # no shuffle
	opt.no_flip = True  # no flip

	data_loader = CreateDataLoader(opt)
	dataset = data_loader.load_data()
	model = create_model(opt)
	visualizer = Visualizer(opt)
	# create website
	web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
	webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
	# test
	avgPSNR = 0.0
	avgSSIM = 0.0
	counter = 0

	target_size = (256, 256)	
	transform = transforms.Compose([
    transforms.Resize(target_size),  # Resize the image to the target size
    transforms.ToTensor()           # Convert the image to a tensor (C, H, W)
])
	path = 'C:/Users/Hieu/OneDrive/Desktop/Kì 1 năm 3/Digital image processing/Mid-term/blurred_sharp/blurred_sharp/resized_sharp'
	for i in os.listdir(path): 
		img_path = os.path.join(path, i)
		# print(img_path) 
		img = Image.open(img_path)
		img = transform(img)
		

	for i, data in enumerate(dataset):
		if i >= opt.how_many:
			break
		counter = i
		model.set_input(data)
		model.test()
		visuals = model.get_current_visuals()
		# Convert visuals['fake_B'] and visuals['real_A'] to tensors with the correct shape
		tensor_fake = torch.tensor(visuals['fake_B']).permute(2, 0, 1).unsqueeze(0).float()  # Generated image

		# Load the corresponding ground truth image from the path directory
		img_path = os.path.join(path, os.listdir(path)[i])  # Ensure order consistency
		ground_truth_img = Image.open(img_path)
		ground_truth_img = transform(ground_truth_img)

		# Convert ground truth image to tensor
		tensor_ground_truth = ground_truth_img.unsqueeze(0).float()

		# Calculate metrics between generated image and ground truth
		avgSSIM += ssim(tensor_fake, tensor_ground_truth)
		avgPSNR += psnr(tensor_fake, tensor_ground_truth, data_range=255.0)

		img_path = model.get_image_paths()
		print('process image... %s' % img_path)
		visualizer.save_images(webpage, visuals, img_path)
		print(f'Image path: {img_path}')

	print(f'Total images processed: {counter}')
		
	avgPSNR /= counter
	avgSSIM /= counter
	print('PSNR = %f, SSIM = %f' %
					  (avgPSNR, avgSSIM))

	webpage.save()
