
from PIL import Image, ImageEnhance
import cv2
import numpy as np

N_RHO_BINS = 7
N_ANGLE_BINS = 12
N_BINS = N_RHO_BINS * N_ANGLE_BINS
BIN_SIZE = 360 // N_ANGLE_BINS
R_INNER = 5.0
R_OUTER = 35.0
K_S = np.arange(3, 8)
sharpness_factor=10
bordersize=3



        
    
def preprocess_image(img_file, sharpness_factor = 10, bordersize = 3):
    im = Image.open(img_file)

    enhancer = ImageEnhance.Sharpness(im)
    im_s_1 = enhancer.enhance(sharpness_factor)
    # plt.imshow(im_s_1, cmap='gray')
    
    (width, height) = (im.width * 2, im.height * 2)
    im_s_1 = im_s_1.resize((width, height))
    image = np.array(im_s_1)
    image = cv2.copyMakeBorder(
        image,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image,(3,3),0)

    (thresh, bw_image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return bw_image

def get_contour_pixels(bw_image):
    contours, _= cv2.findContours(bw_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]

    img2 = bw_image.copy()[:,:,np.newaxis]
    img2 = np.concatenate([img2, img2, img2], axis = 2)

    return contours

def get_cold_features(img_file, approx_poly_factor = 0.01):
    bw_image = preprocess_image(img_file, sharpness_factor, bordersize)
    contours = get_contour_pixels(bw_image)
    
    rho_bins_edges = np.log10(np.linspace(R_INNER, R_OUTER, N_RHO_BINS))
    feature_vectors = np.zeros((len(K_S), N_BINS))
    
    for j, k in enumerate(K_S):
        hist = np.zeros((N_RHO_BINS, N_ANGLE_BINS))
        for cnt in contours:
            epsilon = approx_poly_factor * cv2.arcLength(cnt,True)
            cnt = cv2.approxPolyDP(cnt,epsilon,True)
            n_pixels = len(cnt)
            
            point_1s = np.array([point[0] for point in cnt])
            x1s, y1s = point_1s[:, 0], point_1s[:, 1]
            point_2s = np.array([cnt[(i + k) % n_pixels][0] for i in range(n_pixels)])
            x2s, y2s = point_2s[:, 0], point_2s[:, 1]
            
            thetas = np.degrees(np.arctan2(y2s - y1s, x2s - x1s) + np.pi)
            rhos = np.sqrt((y2s - y1s) ** 2 + (x2s - x1s) ** 2)
            rhos_log_space = np.log10(rhos)
            
            quantized_rhos = np.zeros(rhos.shape, dtype=int)
            for i in range(N_RHO_BINS):
                quantized_rhos += (rhos_log_space < rho_bins_edges[i])
                
            for i, r_bin in enumerate(quantized_rhos):
                theta_bin = int(thetas[i] // BIN_SIZE) % N_ANGLE_BINS
                hist[r_bin - 1, theta_bin] += 1
            
        normalised_hist = hist / hist.sum()
        feature_vectors[j] = normalised_hist.flatten()
        
    return feature_vectors.flatten()







