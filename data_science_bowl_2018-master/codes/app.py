# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:18:47 2019

@author: hemax
"""

from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
from config import *
from predict_gui import *

root = Tk()
root.geometry("550x300+300+150")
root.resizable(width=True, height=True)

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename

def open_img():
    x = openfn()
    img = Image.open(x)
    #img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.pack()
    return img
    
def segmentation():
    from predict_gui import pred_n_plot_test
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

    img = open_img()
    
    
    import time

    start = time.time()

    # Create model configuration in inference mode
    config = KaggleBowlConfig()
    config.GPU_COUNT = 1
    config.IMAGES_PER_GPU = 1
    config.BATCH_SIZE = 1
    config.display()

    # Predict using the last weights in training directory
    model = get_model(config)
    
    result = pred_n_plot_test(model, config, img, save_plots=True)
    
    panel = Label(root, image=result[:,:,0])
    panel.image = result[:,:,0]
    panel.pack()
    

#btn = Button(root, text='Open Image', command=open_img).pack()

btn2 = Button(root, text='Segmentation', command = segmentation).pack()


root.mainloop()