# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:33:55 2019

@author: hemax
"""

import numpy as np 
import pickle
import os
import cv2

predictions_path = r'D:\Ambiente_de_Trabalho\Tese\Code\Mask_RCNN_5thPLACE_Kaggle\data_science_bowl_2018-master\predictions_pickle'


for pkl in os.listdir(predictions_path):

	#read the predictions
	pickle_in = open(os.path.join(predictions_path,pkl), "rb")
	masks = pickle.load(pickle_in)


	#create a labelled image correspondent to the predictions
	prediction = np.zeros(np.shape(masks[:,:,0]))

	for i in range(len(masks[0,0,:])):
		prediction[masks[:,:,i]==1] = 1
        
	aux = pkl    
	aux = aux.replace('.pickle', "")
    
	cv2.imwrite('D:/Ambiente_de_Trabalho/Tese/Code/Mask_RCNN_5thPLACE_Kaggle/data_science_bowl_2018-master/pred_im_13/' + aux + '.tif', prediction * 255.0 )
