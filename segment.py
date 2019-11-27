import numpy as np
import progressbar
import csv
import cv2
from matplotlib import pyplot as plt

sz = 32
count = [20,60,80,80,120]

def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        q = np.true_divide(a, b)
        q[ ~ np.isfinite(q) ] = 0  # -inf inf NaN
    return q

def index_diff(image):
    bgr_sum = np.sum(image, axis=2)

    blues = div0(image[:, :, 0], bgr_sum)
    greens = div0(image[:, :, 1], bgr_sum)
    reds = div0(image[:, :, 2], bgr_sum)

    green_index = 2.0*greens - (reds+blues)
    red_index = reds - greens

    return green_index - red_index

def split_watershed(orig):
    #orig = cv2.imread(filepath)
    green = index_diff(orig)
    gray = cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                    cv2.THRESH_BINARY,55,2)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(thresh,kernel,iterations=3)
    sure_bg[green < 0] = 0

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(thresh,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)
    sure_fg[green < 0] = 0

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(orig,markers)
    #orig[markers == -1] = [255,0,0]
    binary = gray.copy()*0
    binary[markers == -1] = 255
    return split_image(orig,binary)

def split_cannyedge(orig):
    #orig = cv2.imread(filepath)
    green = index_diff(orig)
    gray = cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(orig,100,200)
    edges[green < 0] = 0
    return split_image(orig,edges)

def split_image(gray,binary):
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(gray, contours, -1, 255, 1)
    X = []
    Y = []
    W = []
    H = []
    A = []
    for i,cnt in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect( cnt );
        X.append(x)
        Y.append(y)
        W.append(w)
        H.append(h)
        A.append(w*h)
    crop_img = []
    i=0
    for a,x,y,w,h in sorted(zip(A,X,Y,W,H),reverse=True):
        #if a>5000: continue
        xm = int(x+w/2)
        ym = int(y+h/2)
        if (xm<sz or ym<sz or xm>640-sz or ym>480-sz): continue
        crop_img.append(gray[ym-sz:ym+sz, xm-sz:xm+sz])
        i = i+1
        if i==120: break
    return crop_img
    

wDataWater = []
wDataWater.append(['file_name',	'annotation'])
wDataCanny = []
wDataCanny.append(['file_name',	'annotation'])

with open('../train_files.csv') as csvfile:
    
    readCSV = csv.reader(csvfile, delimiter=',')
    next(readCSV)   #ignore the header row
    
    bar = progressbar.ProgressBar(maxval=896, 
                                    widgets=[progressbar.Bar('=', '[', ']'), ' ', 
                                            progressbar.Percentage()])
    bar.start()
    i = 0
    for row in readCSV:
        # Reading the true labels
        label = row[1]
        file_path = '../Training/'+row[0]
        row = row[0].split('.')
        #row[0] = (row[0]+'_'+str(7)+'.'+row[1])
        #row[1] = label
        #print(row)
        img = cv2.imread(file_path)

        cropped = split_watershed(img)
        for id,im in enumerate(cropped):
            if (i<96):
                if(id==50): break
            else:
                if(id==count[int(label)]): break
            Fname = row[0]+'_'+str(id)+'.'+row[1]
            cv2.imwrite('../Training_water/'+Fname,im)
            wDataWater.append([Fname, label])

        cropped = split_cannyedge(img)
        for id,im in enumerate(cropped):
            if (i<96):
                if(id==50): break
            else:
                if(id==count[int(label)]): break
            Fname = row[0]+'_'+str(id)+'.'+row[1]
            cv2.imwrite('../Training_canny/'+Fname,im)
            wDataCanny.append([Fname, label])
        
        i = i+1
        bar.update(i)
    
    bar.finish()
    
np.savetxt("../train_water.csv", wDataWater, delimiter=",", fmt='%s')
np.savetxt("../train_canny.csv", wDataCanny, delimiter=",", fmt='%s')

