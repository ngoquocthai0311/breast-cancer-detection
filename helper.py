import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

def calculate_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(5,5)) 
    clahe_image = clahe.apply(image)
    return clahe_image


def image_cut(img):
    def return_best_fit_polyfit_value(x, y):
        # This function serves for future improvement 
        Y = []
        best_options = dict()
        flag = False
    
        for i in range(2, 6):

            p = np.polyfit(x,y,i)
            Y = np.polyval(p, x)
            r2squared = r2_score(y, Y)
            best_options[r2squared] = Y
        
        best_r2_score = 0
        if not flag:
            count = 2
            for key in best_options.keys():
                if best_r2_score < key:
                    best_r2_score = key
                    Y = best_options[key]
                count += 1

        return Y
    
    def experiment(img):
        hist,bins=np.histogram(img,bins=256)
        X=bins[0:-1]
        return_image = None
        degree_range = range(2,6)
        for degree in degree_range:
            # Find the polynomial fit
            p=np.polyfit(X,hist,degree)

            # calculate new value y using p(x)
            Y=np.polyval(p,X)

            # Old code
            # dY_abs=np.sort(np.abs(dY))
            # ind=np.argsort(np.abs(dY))
            # d2Y_abs=np.diff(dY_abs)
            # voluem=np.min(d2Y_abs)
            # i=np.argmin(d2Y_abs)

            # 2nd differentiate of new value y
            dY=np.diff(Y)
            dY_abs=np.sort(np.abs(dY))
            d2Y_abs=np.diff(dY_abs)

            # get list sort index and get index that has min value
            ind=np.argsort(np.abs(d2Y_abs))
            i=np.argmin(d2Y_abs)

            # get threshold index value based on index of min value of 2nd differentiate function
            thresh=ind[i]

            # apply threshold value and cut the image
            thresh_img=cv2.threshold(img,thresh,255,cv2.THRESH_BINARY)[1]
            thresh_img = thresh_img.astype(np.uint8)
            (cnts, _) = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if (len(cnts) == 0):
                continue
            segmented = max(cnts, key=cv2.contourArea)
            (x,y,w,h) = cv2.boundingRect(segmented)
            crop_img=img[y:y+h,x:x+w]

            # Get new width, height of cropped image as well as original image
            crop_h, crop_w = crop_img.shape
            origin_h, origin_w = img.shape

            # Check whether the cropped image ís actually cropped that captures fully the ROI. 
            # If the cropped image is not cropped at all then move on.
            if crop_h != origin_w and crop_w != origin_w and crop_h >= 50 / 100 * origin_h and crop_w >= 50 / 100 * origin_w:
                return_image = crop_img
                break

        return return_image

    return experiment(img)

def plot_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def plot_curve(series, hist, bins):
    poly = np.poly1d(series)
    func = np.poly1d(poly)

    new_x = np.linspace(min(bins), max(bins))
    new_y = func(new_x)

    plt.scatter(bins, hist, alpha=0.5)
    plt.plot(new_x, new_y, color='r', lw=4)
    plt.show()

def plot_histogram(gray_img):
    plt.hist(gray_img.ravel(),256,[0,256])
    plt.title('Histogram for gray scale picture')
    plt.show()