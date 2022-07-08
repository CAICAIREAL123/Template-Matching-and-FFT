import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix



def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.
    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.
    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.
    It is recommended you use Hough tools to find these circles in
    the image.
    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.
    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.
    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    
    # convert to gray img
    imgGray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    minR = radii_range[0]-1
    maxR = radii_range[-1]+1
    circlesGray = cv2.HoughCircles(imgGray, cv2.HOUGH_GRADIENT, 1, 50, param1 = 20, param2 = 10, minRadius = minR, maxRadius = maxR)
    if len(circlesGray) == 0:
        print("No detection")
    imgHLS = cv2.cvtColor(img_in, cv2.COLOR_BGR2HLS)
    brightness = []
    xy = []
    for c in circlesGray[0]:
        brightness.append(imgHLS[int(c[1]), int(c[0])][1])
        xy.append((c[0], c[1]))
    index = np.array(brightness).argmax()
    color = ['red', 'yellow', 'green']
    return xy[index], color[index]

def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.
    Args:
        img_in (numpy.array): image containing a traffic light.
    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([10, 50, 50]), np.array([20, 255, 255]))
    mask = cv2.bitwise_and(img_in,img_in, mask= mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(mask,30,400)
    lines_ = cv2.HoughLinesP(edges,0.5,np.pi/180, threshold = 40,minLineLength = 40,maxLineGap = 100)
    lines = []
    for l in lines_:
        x1,y1,x2,y2 = l[0]
        angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle)<48:
            lines.append(l[0])
    lines = np.array(lines)
    x, y  = int(lines[:,[0,2]].mean()), int(lines[:,[1,3]].mean())
    return (x, y)

def template_match(img_orig, img_template, method):
    """Returns the location corresponding to match between original image and provided template.
    Args:
        img_orig (np.array) : numpy array representing 2-D image on which we need to find the template
        img_template: numpy array representing template image which needs to be matched within the original image
        method: corresponds to one of the four metrics used to measure similarity between template and image window
    Returns:
        Co-ordinates of the topmost and leftmost pixel in the result matrix with maximum match
    """
    """Each method is calls for a different metric to determine 
       the degree to which the template matches the original image
       We are required to implement each technique using the 
       sliding window approach.
       Suggestion : For loops in python are notoriously slow
       Can we find a vectorized solution to make it faster?
    """
    #result = np.zeros(
        #(
           # (np.shape(img_orig)[1] - np.shape(img_template)[1] + 1),
            #(np.shape(img_orig)[0] - np.shape(img_template)[0] + 1),
        #),
        #float,
    #)
    #top_left = []
    """Once you have populated the result matrix with the similarity metric corresponding to each overlap, return the topmost and leftmost pixel of
    the matched window from the result matrix. You may look at Open CV and numpy post processing functions to extract location of maximum match"""
    #print(np.shape(img_orig))
    #print(np.shape(img_template))
    #img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    #template = img_template
    #template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
    #print(template)
    img = img_orig
    template = img_template
    index = 0
    minI = 0
    minJ = 0
    imgX, imgY = np.shape(img)
    templateX, templateY = np.shape(template)
    minMethod = 100000000000
    # Sum of squared differences
    if method == "tm_ssd":
        """Your code goes here"""
        #raise NotImplementedError
        for i in range(imgX-templateX):
            for j in range(imgY-templateY):
                index = np.sum((template[:,:] - img[i: i+templateX, j:j+templateY])**2)
                if index<minMethod:
                    minMethod = index
                    minI = i
                    minJ = j
    # Normalized sum of squared differences
    elif method == "tm_nssd":
        """Your code goes here"""
        #raise NotImplementedError
        for i in range(imgX-templateX):
            for j in range(imgY-templateY):
                rootSquareTemplate = np.sqrt(np.sum(np.square(template), axis=0))
                rootSquareImg = np.sqrt(np.sum(np.square(img[i: i+templateX, j:j+templateY]), axis=0))
                index = np.sum((template[:,:]/rootSquareTemplate - img[i: i+templateX, j:j+templateY]/rootSquareImg)**2)
                if index<minMethod:
                    minMethod = index
                    minI = i
                    minJ = j

    # Cross Correlation
    elif method == "tm_ccor":
        """Your code goes here"""
        #raise NotImplementedError
        minMethod = 0
        for i in range(imgX-templateX):
            for j in range(imgY-templateY):
                numerator = np.mean((template[:,:] - template[:,:].mean()) * (img[i: i+templateX, j:j+templateY] - img[i: i+templateX, j:j+templateY].mean()))
                denominator = template[:,:].std() * img[i: i+templateX, j:j+templateY].std()
                if denominator != 0:
                    index = numerator / denominator
                    #print(index)
                    if abs(index)>minMethod:
                        minMethod = index
                        minI = i
                        minJ = j

    # Normalized Cross Correlation
    elif method == "tm_nccor":
        """Your code goes here"""
        #raise NotImplementedError
        minMethod = 0
        for i in range(imgX-templateX):
            for j in range(imgY-templateY):
                rootSuqareTemplate = np.sqrt(np.sum(template**2))
                rootSquareImg = np.sqrt(np.sum(img[i: i+templateX, j:j+templateY]**2))
                numerator = np.mean((template[:,:] - template[:,:].mean()) * (img[i: i+templateX, j:j+templateY] - img[i: i+templateX, j:j+templateY].mean()))
                denominator = template[:,:].std() * img[i: i+templateX, j:j+templateY].std()
                if denominator!=0:
                    index = numerator / denominator
                    index = index/rootSuqareTemplate/rootSquareImg
                    print(index)
                    if abs(index)>minMethod:
                        minMethod = index
                        minI = i
                        minJ = j
        

    else:
        """Your code goes here"""
        # Invalid technique
    #raise NotImplementedError
    return (minJ, minI)

'''Below is the helper code to print images for the report'''
#     cv2.rectangle(img_orig,top_left, bottom_right, 255, 2)
#     plt.subplot(121),plt.imshow(result,cmap = 'gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(img_orig,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(method)
#     plt.show()


def dft(x):
    """Discrete Fourier Transform for 1D signal
    Args:
        x (np.array): 1-dimensional numpy array of shape (n,) representing signal
    Returns:
        y (np.array): 1-dimensional numpy array of shape (n,) representing Fourier Transformed Signal

    """
    x = np.asarray(x, dtype=np.complex_)
    #raise NotImplementedError
    X = len(x)
    n = np.arange(X)
    N = n.reshape((X,1))
    e = np.exp(-2j*np.pi*N*n/X)
    
    return np.dot(e,x)

def idft(x):
    """Inverse Discrete Fourier Transform for 1D signal
    Args:
        x (np.array): 1-dimensional numpy array of shape (n,) representing Fourier-Transformed signal
    Returns:
        y (np.array): 1-dimensional numpy array of shape (n,) representing signal

    """
    x = np.asarray(x, dtype=np.complex_)
    #raise NotImplementedError
    X = len(x)
    n = np.arange(X)
    N = n.reshape((X,1))
    e = np.exp(2j*np.pi*N*n/X)
    e = e/X
    return np.dot(e, x)

def dft2(img):
    """Discrete Fourier Transform for 2D signal
    Args:
        img (np.array): 2-dimensional numpy array of shape (n,n) representing image
    Returns:
        y (np.array): 2-dimensional numpy array of shape (n,n) representing Fourier-Transformed image

    """
    #raise NotImplementedError
    d = np.asarray(img)
    n = np.shape(img)[0]
    x = np.zeros((n,n), dtype = np.complex128)
    for i in range(n):
        for j in range(n):
            tem = 0.0
            for k in range(n):
                for l in range(n):
                    e = np.exp(- 2j * np.pi * ((i * k) / n + (j * l) / n))
                    tem +=d[k, l]*e
            x[i,j] = tem
    return x


def idft2(img):
    """Discrete Fourier Transform for 2D signal
    Args:
        img (np.array): 2-dimensional numpy array of shape (n,n) representing Fourier-Transformed image
    Returns:
        y (np.array): 2-dimensional numpy array of shape (n,n) representing image

    """
    #raise NotImplementedError
    d = np.asarray(img)
    n = np.shape(img)[0]
    x = np.zeros((n,n), dtype = np.complex128)
    for i in range(n):
        for j in range(n):
            tem = 0.0
            for k in range(n):
                for l in range(n):
                    e = np.exp( 2j * np.pi * ((i * k) / n + (j * l) / n))
                    e = e/n/n
                    tem +=d[k, l]*e
            x[i,j] = tem
    return x

def compress_image_fft(img_bgr, threshold_percentage):
    """Return compressed image by converting to fourier domain, thresholding based on threshold percentage, and converting back to fourier domain
    Args:
        img_bgr (np.array): numpy array of shape (n,n,3) representing bgr image
        threshold_percentage (float): between 0 and 1 representing what percentage of Fourier image to keep
    Returns:
        img_compressed (np.array): numpy array of shape (n,n,3) representing compressed image
        compressed_frequency_img (np.array): numpy array of shape (n,n,3) representing the compressed image in the frequency domain

    """
    #raise NotImplementedError
    #print(np.shape(img_bgr))
    sh = np.shape(img_bgr)
    img_compressed = np.zeros(sh, dtype = np.complex128)
    compressed_frequency_img = np.zeros(sh, dtype = np.complex128)
    for channel in range(3):
        temp = img_bgr[:,:,channel]
        temp1 = np.fft.fft2(temp)
        tempSort = np.sort(np.abs(temp1.reshape(-1)))
        thresh = tempSort[int(np.floor((1-threshold_percentage)*len(tempSort)))]
        ind = np.abs(temp1)>thresh
        temp1 = temp1*ind
        #np.place(temp1, abs(temp1) <= threshold_percentage*temp1, [0])
        compressed_frequency_img[:,:,channel] = (temp1)
        img_compressed[:,:,channel] = np.fft.ifft2(temp1)
        #print(np.shape(temp))
    #print(type(img_compressed))
    #print(np.shape(img_compressed))
    
    return img_compressed.real,compressed_frequency_img.real
        
    


def low_pass_filter(img_bgr, r):
    """Return low pass filtered image by keeping a circle of radius r centered on the frequency domain image
    Args:
        img_bgr (np.array): numpy array of shape (n,n,3) representing bgr image
        r (float): radius of low pass circle
    Returns:
        img_low_pass (np.array): numpy array of shape (n,n,3) representing low pass filtered image
        low_pass_frequency_img (np.array): numpy array of shape (n,n,3) representing the low pass filtered image in the frequency domain

    """
    #raise NotImplementedError
    sh = np.shape(img_bgr)
    img_low_pass =  np.zeros(sh, dtype = np.complex128)
    low_pass_frequency_img =  np.zeros(sh, dtype = np.complex128)
    _ = drawCircle(img_bgr.shape[:2],r)
    #print(_)
    img = np.zeros_like(img_bgr,dtype=complex)
    for channel in range(3):
        img[:,:,channel] = np.fft.fftshift(np.fft.fft2(img_bgr[:,:,channel]))
        fft_img_channel  = img[:,:,channel]
        np.place(fft_img_channel, _, [0])
        img_low_pass[:,:,channel] = np.fft.ifft2(np.fft.ifftshift(fft_img_channel))
        low_pass_frequency_img[:,:,channel] = np.fft.ifftshift(fft_img_channel)
    return img_low_pass.real, low_pass_frequency_img.real

def drawCircle(shape, r):
    output = np.zeros(shape,dtype=np.bool)
    center = np.array(output.shape)/2.0
    #print(center)
    for y in range(shape[0]):
        for x in range(shape[1]):
            if (y-center[0])**2 + (x-center[0])**2 <= r **2:
                #print(y,x)
                output[y, x] = False
            else:
                output[y, x] = True
    return output
def filterCircle(ttfCircle, channel):
    temp = np.zeros(channel.shape[:2], dtype = complex)
    temp[ttfCircle] = channel[ttfCircle]
    return temp