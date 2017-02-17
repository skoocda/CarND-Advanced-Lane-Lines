import numpy as np
import cv2
# Functions from classroom

def detect_line(image):
    """
    Applies S-channel threshold and Sobel absolute x-direction gradient threshold.
    Returns binary image.
    """
    s_channel = hls_threshold(image, channel=2, thresh=(160, 255))
    gradx = abs_sobel_thresh(image, orient='x', thresh=(30,200))
    combined = np.zeros_like(s_channel)
    combined[(gradx == 1) | (s_channel == 1)] = 1
    return combined


def abs_sobel_thresh(img, orient='x', sobel_kernel=9, thresh=(0,255)):
    """
    Applies Sobel absolute gradient threshold.
    Returns binary image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Get absolute gradient in x or y using Sobel() function
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0])&(scaled_sobel <= thresh[1])] = 1
    # Return the result
    return binary_output


def mag_thresh(img, sobel_kernel=9, thresh=(0, 255)):
    """
    Applies Sobel gradient magnitude threshold.
    Returns binary image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(np.square(sobelx) + np.square(sobely))
    # Rescale to 8 bit integer
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0])&(gradmag <= thresh[1])] = 1
    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=9, thresh=(0, np.pi / 2)):
    """Applies Sobel gradient direction threshold.
    Returns binary image.
    """
    # Grayscale, as usual
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0])&(absgraddir <= thresh[1])] = 1
    # Return the binary image
    return binary_output


def hls_threshold(img, channel=2, thresh=(0, np.pi / 2)):
    """
    Applies HLS-channel threshold.
    Returns binary image.
    """
    # Extract designated channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    img_channel = hls[:, :, channel]
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(img_channel)
    binary_output[(img_channel >= thresh[0])&(img_channel <= thresh[1])] = 1
    # Return the binary image
    return binary_output
    