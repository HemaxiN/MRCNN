# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:51:01 2019

@author: hemaxi
"""

import PIL
import tkinter
import cv2

from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
 
from skimage.color import label2rgb
from config import *
from model import log
from train import train_validation_split, KaggleDataset

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
from skimage.measure import find_contours
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import dilation, erosion


import random
import pandas as pd
from metrics import mean_iou
from tqdm import tqdm
import os
import numpy as np

from predict_gui import get_model, pred_n_plot_test

import time

start = time.time()

def select_image():
	# grab a reference to the image panels
	global panelA, panelB
 
	# open a file chooser dialog and allow the user to select an input
	# image
	path = filedialog.askopenfilename()
	
# ensure a file path was selected
	if len(path) > 0:
		# load the image from disk, convert it to grayscale, and detect
		# edges in it
		image = cv2.imread(path)
		image = cv2.resize(image, (1024, 1024))

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		edged = cv2.Canny(gray, 50, 100)
        
      # Create model configuration in inference mode
		config = KaggleBowlConfig()
		config.GPU_COUNT = 1
		config.IMAGES_PER_GPU = 1
		config.BATCH_SIZE = 1
		config.display()
		model = get_model(config)      
		result = pred_n_plot_test(model, config, image, save_plots=True )                
        
		edged = label2rgb(result, bg_label = 0, bg_color = (1,1,1))
		 
		cv2.imwrite(r'C:\Users\hemax\Desktop\hello\oi2.png', edged)      
  		
 
		# OpenCV represents images in BGR order; however PIL represents
		# images in RGB order, so we need to swap the channels
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
		# convert the images to PIL format...
		image = Image.fromarray(image)
		edged = np.asarray(edged*255, np.uint8)        
		edged = Image.fromarray(edged)
 
		# ...and then to ImageTk format
		image = ImageTk.PhotoImage(image)
		edged = ImageTk.PhotoImage(edged)
        
# if the panels are None, initialize them
		if panelA is None or panelB is None:
			# the first panel will store our original image
			panelA = Label(image=image)
			panelA.image = image
			panelA.pack(side="left", padx=10, pady=10)
 
			# while the second panel will store the edge map
			panelB = Label(image=edged)
			panelB.image = edged
			panelB.pack(side="right", padx=10, pady=10)
			print('Elapsed time', ((time.time() - start)/60), 'minutes')
		# otherwise, update the image panels
		else:
			# update the pannels
			panelA.configure(image=image)
			panelB.configure(image=edged)
			panelA.image = image
			panelB.image = edged
        
        


# initialize the window toolkit along with the two image panels
root = Tk()
panelA = None
panelB = None
 
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="no", padx="10", pady="10")
 
# kick off the GUI
root.mainloop()