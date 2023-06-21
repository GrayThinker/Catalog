"""
Mask generation

Approach one:
 - Catalog images generally have a white background. We can confirm this by checking the colors at the edge of the image.
 Then we can set the alpha of any pixel that is white to 0 to exclude in the mask.

Approach two:
 - Using edge detection we can get a rough estimate of the roi of the image. This is unlikely to get "holes"

Approach three:
 - Hopefully some model exists out there that is good at extracting object that are front and center on a plane background.


"""

"""

Get largest contor area
find other contors that are min x% of largest contour area
Find average color of area outside largest contour (does it vary a lot)
Check for smaller contour who's internal area average color is very similar to external
Set the inside of those to be excluded in the mask


for smaller contour (exclude larger contour)
 -mode escape if the most common is the foreground
 -mean escape if the average is the foreground 

"""


"""
Use rekognition to find text bounding box so it can be excluded from the holes calculation
When user shades something as foreground or background then exclude or include all
contours that intersect with that stroke.
Will need undo button.

Might use rekognition image properties to get background color.


if stroke completely within a contour and there are no other contour then apply
if contour is completely within stroke then apply
"""

import numpy as np
import cv2

from matplotlib import pyplot as plt
from scipy import stats

def first():
    img = cv2.imread('JBL.jpg')
    assert img is not None, "file could not be read, check with os.path.exists()"
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (50,50,450,290)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    plt.imshow(img),plt.colorbar(),plt.show()


def find_largest_contour(image):
    """
    This function finds all the contours in an image and return the largest
    contour area.
    :param image: a binary image
    """
    image = image.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        image,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    largest_contour = max(contours, key=cv2.contourArea)
    l_cont_area = cv2.contourArea(largest_contour)
    # print(l_cont_area)
    larger_contours = [cv2.contourArea(cont)/l_cont_area * 100 for cont in contours]
    larger_contours.sort(reverse=True)
    # print(larger_contours)
    out = [cont for cont in contours if cv2.contourArea(cont)/l_cont_area * 100 > 0]
    return out, hierarchy

def show(name, image):
    """
    A simple function to visualize OpenCV images on screen.
    :param name: a string signifying the imshow() window name
    :param image: NumPy image to show 
    """
    cv2.imshow(name, image)
    cv2.waitKey(0)

def remove(arr, value):
    index = 0
    for i, val in enumerate(arr):
        if np.array_equal(val, value):
            arr.pop(i)
            return arr
    return arr

def get_valid_contours(image, color_image):
    # need to add minimum contour size
    # need to identify text location to exclude from threshold
    image = image.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        image,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = list()

    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour_area = cv2.contourArea(largest_contour)
    valid_contours.append(largest_contour)

    temp = np.zeros_like(image)
    cv2.drawContours(temp, [largest_contour], 0, color=255, thickness=-1)
    pts = np.where(temp==0)
    bkg = color_image[pts[0], pts[1]]
    print(stats.mode(bkg))
    # show('large contour', temp)
    bkg_mode = stats.mode(bkg).mode[0]
    # print([cv2.contourArea(c)/largest_contour_area * 100 for c in contours])
    # print(list(contours).index(largest_contour))
    contours = list(contours)
    contours = remove(contours, largest_contour)
    # print([cv2.contourArea(c)/largest_contour_area * 100 for c in contours])
    for i in range(len(contours)):
        # TODO: check if inside largest contour
        temp = np.zeros_like(image)
        # show('f', cimg)
        cv2.drawContours(temp, contours, i, color=255, thickness=-1)
        # cv2.drawContours(temp, contours, i, color=255, thickness=-1)
        pts = np.where(temp==255)
        hole = color_image[pts[0], pts[1]]
        # print(stats.mode(hole).mode)
        # show('small contour', temp)
        if np.array_equal(stats.mode(hole).mode[0], bkg_mode):
            valid_contours.append(contours[i])
    
    # print(len([cv2.contourArea(contour)/largest_contour_area * 100 for contour in contours]))
    # print(len([cv2.contourArea(contour)/largest_contour_area * 100 for contour in valid_contours]))

    return valid_contours

def extract_foreground():
    """
    TODO: need to remove white outline
    """
    file_name = "resources\\chair"
    image = cv2.imread(f'{file_name}.jpg')
    show('Input image', image)
    # blur the image to smmooth out the edges a bit, also reduces a bit of noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # convert the image to grayscale 
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # apply thresholding to conver the image to binary format
    # after this operation all the pixels below 200 value will be 0...
    # and all th pixels above 200 will be 255
    ret, gray = cv2.threshold(gray, 250 , 255, cv2.CHAIN_APPROX_NONE)
    # If image doesn't have white background/ pixels close > 240 then the entire image get threshold to 0 and no contours detected.
    
    # find the largest contour area in the image
    # contour, hierarchy = find_largest_contour(gray)
    contour = get_valid_contours(gray, image)

    image_contour = np.copy(image)
    cv2.drawContours(image_contour, contour, -1, (0, 255, 0), 1, cv2.LINE_AA)
    # show('Contour', image_contour)

    # # outside largest contout:
    # ximg = np.zeros_like(image)
    # l_cont = max(contour, key=cv2.contourArea)
    # cv2.drawContours(ximg, [l_cont], 0, color=255, thickness=-1)
    # pts = np.where(ximg==0)
    # bkg = image[pts[0], pts[1]]
    # bkg_mode = stats.mode(bkg).mode[0]
    # bkg_mean = np.mean(bkg, axis=0)
    # print(f"mean: {bkg_mean}")
    # print(f"mode: {bkg_mode}")
    # # show('max', ximg)


    # for i in range(len(contour)):
    #     cimg = np.zeros_like(image)
    #     # show('f', cimg)
    #     cv2.drawContours(cimg, contour, i, color=255, thickness=-1)
    #     pts = np.where(cimg==255)
    #     # show('l', cimg)
    #     # print(image[pts[0], pts[1]])

    # create a black `mask` the same size as the original grayscale image 
    mask = np.zeros_like(gray)
    # fill the new mask with the shape of the largest contour
    # all the pixels inside that area will be white 
    cv2.drawContours(mask, [contour[0]], 0, color=255, thickness=-1)
    cv2.drawContours(mask, [contour[0]], 0, color=0, thickness=5)
    # show('final large contour', mask)
    # cv2.fillPoly(mask, contour[0], 255)
    cv2.drawContours(mask, contour[1:], -1, color=0, thickness=-1)
    cv2.drawContours(mask, contour[1:], -1, color=0, thickness=5)
    # show('final with small', mask)
    # create a copy of the current mask
    res_mask = np.copy(mask)
    res_mask[mask == 0] = cv2.GC_BGD # obvious background pixels
    res_mask[mask == 255] = cv2.GC_PR_BGD # probable background pixels
    res_mask[mask == 255] = cv2.GC_FGD # obvious foreground pixels

    # create a mask for obvious and probable foreground pixels
    # all the obvious foreground pixels will be white and...
    # ... all the probable foreground pixels will be black
    mask2 = np.where(
        (res_mask == cv2.GC_FGD) | (res_mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype('uint8')

    # create `new_mask3d` from `mask2` but with 3 dimensions instead of 2
    new_mask3d = np.repeat(mask2[:, :, np.newaxis], 3, axis=2)
    mask3d = new_mask3d
    mask3d[new_mask3d > 0] = 255.0
    mask3d[mask3d > 255] = 255.0
    # apply Gaussian blurring to smoothen out the edges a bit
    # `mask3d` is the final foreground mask (not extracted foreground image)
    mask3d = cv2.GaussianBlur(mask3d, (5, 5), 0)
    show('Foreground mask', mask3d)

    # create the foreground image by zeroing out the pixels where `mask2`...
    # ... has black pixels
    # mask2 = cv2.GaussianBlur(mask2, (5, 5), 0)
    foreground = np.copy(image).astype(float)
    alpha_channel = (np.ones(foreground.shape[0:2]) * 255).astype(np.uint8)
    alpha_channel[mask2 == 0] = 0
    print("alpha after")
    print(alpha_channel[0:3])
    alpha_img = np.dstack((foreground, alpha_channel))
    print("alpha img")
    print(alpha_img[0:3])
    # alpha_img[mask2 == 0] = 0
    show('Foreground', alpha_img.astype(np.uint8))

    # save_name = 'JBL-out'
    # cv2.imwrite(f"outputs/{save_name}_foreground.png", foreground)
    # cv2.imwrite(f"outputs/{save_name}_foreground_mask.png", mask3d)
    # cv2.imwrite(f"out.jpg", image_contour)
    cv2.imwrite(f"{file_name}-out.png", alpha_img)
    print("done")

extract_foreground()