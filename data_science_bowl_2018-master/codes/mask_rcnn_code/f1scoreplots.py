# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:36:43 2019

@author: hemaxi
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


pandasfilenames = [r'D:\Ambiente_de_Trabalho\Tese\DeepLearningNucleiSegmentation\pandas_tables\monday_3channel_final.pickle',
                   r'D:\Ambiente_de_Trabalho\Tese\DeepLearningNucleiSegmentation\pandas_tables\monday_ORUNET_final.pickle',
                   r'D:\Ambiente_de_Trabalho\Tese\DeepLearningNucleiSegmentation\pandas_tables\monday_Watershed_final.pickle',
                   r'D:\Ambiente_de_Trabalho\Tese\DeepLearningNucleiSegmentation\pandas_tables\monday_ContourWmap_final.pickle']

fig, ax = plt.subplots()
def f1plot (namelist):
	color = ['r', 'b', 'g', 'm', 'y', 'c', 'k']

	for i in range(len(namelist)):
        
		df = pd.read_pickle(namelist[i])        
		df = df[df['Model'] == 'Model13']  
		df = df.groupby(df.index).agg({'F1': np.mean, 'Jaccard': np.mean, 
		                                     'TP': np.sum, 'FP': np.sum, 'FN': np.sum, 'Precision': np.mean, 
		                                     'Recall': np.mean, 'Splits': np.sum, 'Merges': np.sum})
		x = df.index.values
		y = df["F1"].values
		yp = None
		xi = np.linspace(x[0], x[-1], 500)
		yi = np.interp(xi, x, y, yp)
		ax.plot(x, y, color[i]+'o', xi, yi, color[i]+'-')

	ax.set_title('F1-Score vs IoU Threshold for different models')
	ax.set_xlabel('IoU Threshold')
	ax.set_ylabel('F1 Score')
	plt.show()

leg = ['3 Channel Mask - UNET', 'Original UNET', 'Threshold + Watershed', 'Contour Weight Map', 'Mask RCNN']
f1plot(pandasfilenames)

plt.legend([leg[0],'interpolated', leg[1], 'interpolated', leg[2], 'interpolated', leg[3], 'interpolated', leg[4], 'interpolated'], loc = 'lower left')

