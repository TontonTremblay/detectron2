import glob 
import cv2 
import pickle 
import numpy as np

path = 'output/*.png'

imgs = []
data = []

for i_img, img_path in enumerate(glob.glob(path)):
	if '_p' in img_path:
		continue
	img = cv2.imread(img_path)
	data = pickle.load( open( img_path.replace('png','p'), "rb" ) )
	print(data.pred_classes)
	print(data.pred_masks.shape)
	for i_pred in range(len(data.pred_classes)):
		if data.pred_classes[i_pred] in [73]:
			img_alpha = np.ones((480, 640,4))*0
			img_alpha[:,:,:3] = img
			img_alpha[data.pred_masks[i_pred],-1] = 255
			# img_alpha[data.pred_masks[i_pred]*-1,:] = 0

			# print(img.shape)
			# print(img_alpha.shape)

			# img[data.pred_masks[0]] = [1,1,1]
			cv2.imwrite(f'output_colmap/{str(i_img).zfill(4)}.png',img_alpha)
			# raise()