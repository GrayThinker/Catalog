from scipy import stats
from PIL import Image
import numpy as np
import cv2


def np_remove(arr, value):
    for i, item in enumerate(arr):
        if np.array_equal(item, value):
            arr.pop(i)
            return arr, True
    return arr, False


def show(image):
    """
    A simple function to visualize OpenCV images on screen.
    :param image: NumPy image to show 
    """
    cv2.imshow("Image", image)
    cv2.waitKey(0)


def get_valid_contours(gray, color):
    gray = gray.astype(np.uint8)
    contours, _ = cv2.findContours(
        gray, 
        cv2.RETR_TREE, 
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = list()
    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour_area = cv2.contourArea(largest_contour)
    valid_contours.append(largest_contour)

    temp = np.zeros_like(gray)
    cv2.drawContours(temp, [largest_contour], 0, color=255, thickness=-1)
    bkg_pts = np.where(temp==0)
    bkg = color[bkg_pts[0], bkg_pts[1]]
    bkg_color = stats.mode(bkg).mode[0]

    contours = np_remove(list(contours), largest_contour)

    for i in range(len(contours)):
        # TODO: check if inside largest contour
        temp = np.zeros_like(gray)

        cv2.drawContours(temp, contours, i, color=255, thickness=-1)
        contour_pts = np.where(temp==255)
        contour_color_pts = color[contour_pts[0], contour_pts[1]]
        contour_color = stats.mode(contour_color_pts).mode[0]
        
        # remove holes
        if np.array_equal(contour_color, bkg_color):
            valid_contours.append(contours[i])
    return valid_contours


def extract_foreground(image, edge=5, threshold=250, verbose=False):
    if verbose:
        show(image)

    # blur the image to smmooth out the edges a bit, also reduces a bit of noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)   
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # apply thresholding to convert the image to binary format
    # after this operation all the pixels below threshold will be 0
    # and all th pixels above threshold will be 255
    _, gray = cv2.threshold(gray, threshold, 255, cv2.CHAIN_APPROX_NONE)
    contours = get_valid_contours(gray, image) #TODO: test blurred

    mask = np.zeros_like(gray)

    # fill the new mask with the shape of the largest contour 
    cv2.drawContours(mask, [contours[0]], 0, color=255, thickness=-1)
    cv2.drawContours(mask, [contours[0]], 0, color=0, thickness=edge)

    # fill mask with holes
    cv2.drawContours(mask, contours[1:], -1, color=0, thickness=-1)
    cv2.drawContours(mask, contours[1:], -1, color=0, thickness=5)

    # create a copy of the current mask
    res_mask = np.copy(mask)
    res_mask[mask == 0] = cv2.GC_BGD # obvious background pixels
    res_mask[mask == 255] = cv2.GC_PR_BGD # probable background pixels
    res_mask[mask == 255] = cv2.GC_FGD # obvious foreground pixels

    # create a mask for obvious and probable foreground pixels
    # all the obvious foreground pixels will be white and...
    # ... all the probable foreground pixels will be black
    mask = np.where(
        (res_mask == cv2.GC_FGD) | (res_mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype('uint8')

    foreground = np.copy(image).astype(float)
    alpha_channel = (np.ones(foreground.shape[0:2]) * 255).astype(np.uint8)
    alpha_channel[mask == 0] = 0
    alpha_img = np.dstack((foreground, alpha_channel))

    if verbose:
        show(alpha_img.astype(np.uint8))
    
    return alpha_img


if __name__ == '__main__':
    extract_foreground(Image.open("resources\\chair.jpg"))
