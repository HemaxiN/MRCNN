# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:29:06 2019

@author: hemax
"""

import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.segmentation import clear_border
import skimage

# =============================================================================
# predictions_path = r'C:/Users/hemax/Desktop/predictions_clones'
# images_path = r'D:/Ambiente_de_Trabalho/Tese/Images/CellImages/Blue_Channel'
# =============================================================================

# =============================================================================
# 
# predictions_path = r'C:/Users/hemax/Desktop/Mock/Predictions'
# images_path = r'C:/Users/hemax/Desktop/Mock/Blue Channel'
# =============================================================================


# =============================================================================
# predictions_path = r'C:\Users\hemax\Desktop\Imagens_Membrana\predictions\CHO'
# images_path = r'C:\Users\hemax\Desktop\Imagens_Membrana\CHO_blue'
# 
# =============================================================================

predictions_path = r'C:\Users\hemax\Desktop\Nova pasta\pred'
images_path = r'C:\Users\hemax\Desktop\Nova pasta'


# =============================================================================
# predictions_path = r'D:\Ambiente_de_Trabalho\Tese\SF\predictions'
# images_path = r'D:\Ambiente_de_Trabalho\Tese\Images\CellImages\Blue_Channel_SF'
# =============================================================================

# =============================================================================
# predictions_path = r'D:/Ambiente_de_Trabalho/Tese/Code/Mask_RCNN_5thPLACE_Kaggle/data_science_bowl_2018-master/codes/mask_rcnn_code/predictions/1'
# images_path = r'D:/Ambiente_de_Trabalho/Tese/NCV/1/Images'
# =============================================================================

for pkl, img in zip(os.listdir(predictions_path), os.listdir(images_path)):
	pickle_in = open(os.path.join(predictions_path, pkl), "rb")
	masks = pickle.load(pickle_in)

	prediction = np.zeros(np.shape(masks[:,:,0]))

	for i in range(len(masks[0,0,:])):
		prediction[masks[:,:,i]==1] = i + 1

	masks_labelled = prediction.astype(np.int64)
	masks = skimage.morphology.remove_small_objects(masks_labelled, min_size = 20)  

	prediction = masks

	plt.rcParams['figure.figsize'] = (10,10)     

	fig = plt.figure()

	ax1 = fig.add_subplot(121)
	img_aux = cv2.imread(os.path.join(images_path,img))
	plt.imshow(img_aux, cmap = 'gray')
	ax1.set_title('Original Image')

	ax2 = fig.add_subplot(122)
	I = label2rgb(prediction, bg_label =0, bg_color=(1,1,1))
	plt.imshow(I)
	ax2.set_title('Prediction ' + img)

	#fig.savefig(r'C:/Users/hemax/Desktop/' + img)
	#plt.close()    
	a = prediction
	a[a!=0] = 1   
	#cv2.imwrite(os.path.join(r'D:\Ambiente_de_Trabalho\Tese\Code\Cell_Cycle_Test_on_New_Images\MIP_AGAIN\grayscale\masks', img), a*255)  
	  
    