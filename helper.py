import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

def calculate_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=0.02, tileGridSize=(8,8)) 
    clahe_image = clahe.apply(image)
    return clahe_image

def resize_image(img, width = 227, height = 227):
    w, h = img.shape
    resized_image = []
    if w <= width or h <= height:
        resized_image = cv2.resize(img, (width, height))

    if w > width or h > height:
        resized_image = cv2.resize(img, (width, height), interpolation = cv2.INTER_LINEAR)

    return resized_image        

def image_cut(img, tracking = False, hasBroken = False):

    def tracking_image(thresh_img, img):
        plot_image(thresh_img)
        plot_image(img)

    def cropped_image_filter(original_img, cropped_img, min_width = 20/100, min_height = 20/100):
        crop_h, crop_w = cropped_img.shape
        origin_h, origin_w = original_img.shape
        if crop_h >= min_height * origin_h and crop_w >= min_width * origin_w:
            return True

        return False

    def second_cropped_image_filter(list_of_images, original_img):
        if len(list_of_images) == 0:
            return []
        # Assume the desirable cropped img is alwasy has the smallest polynomial degree
        filtered_image = list_of_images[0]
        better_cropped_images = []
        # Case when the images cut always stay the same. Accept the image.
        for image in list_of_images:
            crop_h, crop_w = image.shape
            origin_h, origin_w = original_img.shape

            # Case when the list of images has better cropped length. 
            if crop_h != origin_h and crop_w != origin_w:
                better_cropped_images.append(image)
                break 
        
        if len(better_cropped_images) != 0:
            best_image = better_cropped_images[0]
            for image in better_cropped_images:
                
                crop_h, crop_w = image.shape
                best_h, best_w = best_image.shape

                # Case when the list of images has better cropped length. 
                if crop_h > best_h and crop_w > best_w:
                    best_image = image

            filtered_image = best_image
            
        return filtered_image
    
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
    
    def experiment(img, tracking):
        # hist,bins=np.histogram(img,bins=256)
        # X=bins[0:-1]
        return_images = []
        degree_range = range(2,15)
        for degree in degree_range:
            thresh_img = get_thresh_image(img, degree, (not hasBroken))
            (cnts, _) = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if (len(cnts) == 0):
                if (tracking):
                    print(degree)
                    print('cnts equal zero')
                continue
            segmented = max(cnts, key=cv2.contourArea)
            (x,y,w,h) = cv2.boundingRect(segmented)
            crop_img=img[y:y+h,x:x+w]

            # If tracking is needed, plot thresh_img and crop_img in each iteration.
            if tracking:
                print(degree)
                tracking_image(thresh_img, crop_img)

            # Check whether the cropped image ís actually cropped that captures fully the ROI. 
            # If the cropped image is not cropped at all then move on.
            if cropped_image_filter(img, crop_img):
                return_images.append(crop_img)
        
        return return_images


    if not hasBroken:
        crop_images = experiment(img, tracking)
    else:
        crop_images = experiment(img, tracking)

    result_image = second_cropped_image_filter(crop_images, img)

    return result_image

def get_thresh_image(img, degree, include_otsu = True):
    hist,bins=np.histogram(img,bins=256)
    bins=bins[0:-1]

    # Find the polynomial fit
    p=np.polyfit(bins,hist,degree)

    # calculate new value y using p(x)
    Y=np.polyval(p,bins)

    # 2nd differentiate of new value y
    dY=np.diff(Y)
    dY_abs=np.sort(np.abs(dY))
    d2Y_abs=np.diff(dY_abs)

    # get list sort index and get index that has min value
    ind=np.argsort(np.abs(d2Y_abs))
    i=np.argmin(d2Y_abs)

    # get threshold index value based on index of min value of 2nd differentiate function
    thresh=ind[i]
    thresh_img = None
    # apply threshold value and cut the image
    if include_otsu:
        thresh_img=cv2.threshold(img,thresh,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    else:
        thresh_img=cv2.threshold(img,thresh,255,cv2.THRESH_BINARY)[1]

    thresh_img = thresh_img.astype(np.uint8)
    return thresh_img
    

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


def convert_to_rgb(gray_images):
    rgb_imgs = []
    for img in gray_images:
        rgb_imgs.append(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB))

    return rgb_imgs

def display_img_in_bulk(img, time):
    cv2.imshow('Window', img)

    key = cv2.waitKey(time)
    if key == 27:#if ESC is pressed, exit loop
        cv2.destroyAllWindows()
