# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:51:01 2019

@author: hemaxi
"""

import PIL
import Tkinter
import cv2

from Tkinter import *
from PIL import Image
from PIL import ImageTk
import tkFileDialog
import cv2
 
def select_image():
	# grab a reference to the image panels
	global panelA, panelB
 
	# open a file chooser dialog and allow the user to select an input
	# image
	path = tkFileDialog.askopenfilename()