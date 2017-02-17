import numpy as np
import cv2
import matplotlib.pyplot as plt
# Functions from friends on slack + atlassian

def pns_image(image, title='', filename='', gray=False):
    """Plots 'n' saves image."""
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title(title)
    if gray: plt.imshow(image, cmap='Greys_r')
    else: plt.imshow(image)
    if len(filename)>0: plt.savefig(filename + '.jpg')

def add_polygon(image, points):
    """Adds a green polygon to the image"""
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, np.int32([points]), (0,255, 0))
    return cv2.addWeighted(image, 1, mask, 0.3, 0)

