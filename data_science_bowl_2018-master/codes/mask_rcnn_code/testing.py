# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:50:02 2019

@author: hemaxi
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

pickle_in = open('D:\Ambiente_de_Trabalho\Tese\Code\Mask_RCNN_5thPLACE_Kaggle\data_science_bowl_2018-master\predictions_clones', "rb")
masks = pickle.load(pickle_in)

aux = np.zeros((1040, 1388))

for i in range(len(masks[0,0,:])):
    aux[masks[:,:,i]==1] = i+1

#labelled masks
plt.imshow(aux)

# =============================================================================
# path = r'D:\Ambiente_de_Trabalho\Tese\Code\Mask_RCNN_5thPLACE_Kaggle\data_science_bowl_2018-master\predictions'
# masks_path = r'D:\Ambiente_de_Trabalho\Tese\Code\Mask_RCNN_5thPLACE_Kaggle\data_science_bowl_2018-master\masks'
# 
# for pkl, gt in zip(os.listdir(path), os.listdir(masks_path)):
#     print(pkl)
#     print(gt)
#     
# 
# =============================================================================
