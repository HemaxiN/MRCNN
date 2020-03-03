from utils import predict
import os
from utils import f1score_plot
import matplotlib.pyplot as plt
from utils import roccurve
import pickle
import numpy as np
from skimage.measure import label
import pandas as pd
import cv2
from utils import evaluation_metrics


nb = '1'
#path to pickle files created by "predict" on maskrcnn folder (codes)
predictions_path = os.path.join(r'D:\Ambiente_de_Trabalho\Tese\Code\Mask_RCNN_5thPLACE_Kaggle\data_science_bowl_2018-master\codes\mask_rcnn_code\predictions', nb)
masks_path = os.path.join(r'D:\Ambiente_de_Trabalho\Tese\NCV', nb, 'Masks')

#predictions_path = r'C:\Users\hemax\Desktop\predictions_clones'
#masks_path = r'C:\Users\hemax\Desktop\Clones\Clones_Masks_Final'
pandasfilepath = 'maskrcnn.pickle'

#predict.create_new_table(pandasfilepath)

results = pd.DataFrame(columns=["Image", "Threshold", "F1", "Jaccard", "TP", "FP", "FN", "Precision", "Recall",
                                    "Splits", "Merges"])

predictions = []

for pkl, gt in zip(os.listdir(predictions_path), os.listdir(masks_path)):

	#read the predictions
	pickle_in = open(os.path.join(predictions_path,pkl), "rb")
	masks = pickle.load(pickle_in)

	#create a labelled image correspondent to the predictions
	prediction = np.zeros(np.shape(masks[:,:,0]))

	for i in range(len(masks[0,0,:])):
		prediction[masks[:,:,i]==1] = i + 1

	ground_truth = cv2.imread(os.path.join(masks_path, gt), cv2.IMREAD_GRAYSCALE)
	ground_truth = label(ground_truth)
    
	predictions.append(prediction)    

	results = evaluation_metrics.compute_results(
        ground_truth,
        prediction,
        results,
        gt)


average_performance = results.groupby("Threshold").agg({'F1': np.mean, 'Jaccard': np.mean, 
                                     'TP': np.sum, 'FP': np.sum, 'FN': np.sum, 'Precision': np.mean, 
                                     'Recall': np.mean, 'Splits': np.sum, 'Merges': np.sum})
new_col= ['Model'+ nb]*10   
average_performance.insert(loc=0, column="Model", value=new_col)


#read the final results table
aux_results = pd.read_pickle(pandasfilepath)

#add the results of this model    
aux_results = pd.concat([aux_results, average_performance])

#save the results
aux_results.to_pickle(pandasfilepath)

