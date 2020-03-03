# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:58:40 2019

@author: hemax
"""

from skimage.color import label2rgb

name = [0,1,2,3,4,5,6,7,8,9]
	
fig = plt.figure() #create a new figure

ax1 = fig.add_subplot(251)
I = predictions[1] 
I = label2rgb(I, bg_label =0, bg_color=(1,1,1))
ax1.imshow(I)
ax1.set_title(name[1])

ax2 = fig.add_subplot(252)

I = predictions[2] 
I = label2rgb(I, bg_label =0, bg_color=(1,1,1))
ax2.imshow(I)
ax2.set_title(name[2])

ax3 = fig.add_subplot(253)

I = predictions[3] 
I = label2rgb(I, bg_label =0, bg_color=(1,1,1))
ax3.imshow(I)
ax3.set_title(name[3])

ax4 = fig.add_subplot(254)

I = predictions[4] 
I = label2rgb(I, bg_label =0, bg_color=(1,1,1))
ax4.imshow(I)
ax4.set_title(name[4])

ax5 = fig.add_subplot(255)

I = predictions[5] 
I = label2rgb(I, bg_label =0, bg_color=(1,1,1))
ax5.imshow(I)
ax5.set_title(name[5])

ax6 = fig.add_subplot(256)

I = predictions[6] 
I = label2rgb(I, bg_label =0, bg_color=(1,1,1))
ax6.imshow(I)
ax6.set_title(name[6])

ax7 = fig.add_subplot(257)

I = predictions[7] 
I = label2rgb(I, bg_label =0, bg_color=(1,1,1))
ax7.imshow(I)
ax7.set_title(name[7])

ax8 = fig.add_subplot(258)

I = predictions[0]
I = label2rgb(I, bg_label =0, bg_color=(1,1,1))
ax8.imshow(I)
ax8.set_title(name[0])

ax9 = fig.add_subplot(259)

I = predictions[8] 
I = label2rgb(I, bg_label =0, bg_color=(1,1,1))
ax9.imshow(I)
ax9.set_title(name[8])

ax10 = fig.add_subplot(2,5,10)

I = predictions[9] 
I = label2rgb(I, bg_label =0, bg_color=(1,1,1))
ax10.imshow(I)
ax10.set_title(name[9])