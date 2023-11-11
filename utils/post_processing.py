import cv2
import numpy as np


def noise_remove(im):
    kernel = np.ones((5, 5), np.uint8)
    im_re = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(
        im_re, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # calculate points for each contour

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            cv2.fillPoly(im_re, pts=[cnt], color=(0))
    return im_re


def post_processing(outputs_classification, output_lungs, output_infected):
    output_infected = noise_remove(output_infected)
    output_lungs = noise_remove(output_lungs)
    output_infected = cv2.bitwise_and(output_infected, output_lungs, mask=None)

    return outputs_classification, output_lungs, output_infected
