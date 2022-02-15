import glob 
import cv2 
import pickle 
import numpy as np
import subprocess

path = 'output/*_p.png'

imgs = []
data = []

subprocess.call(['rm','-rf','output/train'])
subprocess.call(['rm','-rf','output/colmap'])

subprocess.call(['mkdir','output/train'])
subprocess.call(['mkdir','output/colmap'])

for i_img, img_path in enumerate(sorted(glob.glob(path))):
	# if '_p' in img_path:
	# 	continue
	img_path = img_path.replace('_p','')
	print(i_img,img_path)

	img = cv2.imread(img_path)
	data = pickle.load( open( img_path.replace('png','p'), "rb" ) )
	print(data.pred_classes)
	# print(data.pred_masks.shape)
	img_alpha = np.ones((480, 640,4))*0
	img_alpha[:,:,:3] = img
	for i_pred in range(len(data.pred_classes)):
		# raise()
		# if data.pred_classes[i_pred] in [35]: # baseball glove	
		print(data.pred_masks[i_pred].sum() )
		if data.pred_classes[i_pred] in [73] and data.pred_masks[i_pred].sum().data > 20000: # book
			img_alpha[data.pred_masks[i_pred],-1] = 255
			# img_alpha[data.pred_masks[i_pred],-1] = img[data.pred_masks[i_pred]]
			# img_alpha[data.pred_masks[i_pred]*-1,:] = 0

			# print(img.shape)
			# print(img_alpha.shape)

			# img[data.pred_masks[0]] = [1,1,1]
	if not np.sum(img_alpha[:,:,-1]) == 0:
		# img_alpha[:,:,-1] = img
		mask = img_alpha[:,:,-1]
		# print(mask.shape)
		# raise()
		im = np.ones((480, 640,3))*0
		im[mask>0] = img[mask>0]
		
		imfinal = np.ones((480, 640,4))*0
		imfinal[:,:,:3] = im
		print(imfinal[mask>0].shape)
		# raise()
		# imfinal[mask>0] =
		# raise()
		cv2.imwrite(f'output/train/{str(i_img).zfill(4)}.png',img_alpha)
		cv2.imwrite(f'output/colmap/{str(i_img).zfill(4)}.png',im)
		
		# raise()
			# cv2.imshow(f'{str(i_img).zfill(4)}',img_alpha)