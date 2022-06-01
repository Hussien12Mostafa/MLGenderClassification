from hinge_feature_extraction import *
import numpy as np
import os
from tqdm import tqdm
import time
import sys
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import *
from os import listdir
from os.path import isfile, join

# Load the image 
try : 
    TEST_DIRECTORY   = sys.argv[1]
    OUTPUT_DIRECTORY = sys.argv[2]
    test_files   = [ f for f in listdir(TEST_DIRECTORY) if isfile(join(TEST_DIRECTORY,f)) ]
    print(' - Test files : loaded successfully')
    print("-----------------------------------------------------")
except: 
    print("Error: No test directory found")
    sys.exit()

try: 
    os.mkdir(OUTPUT_DIRECTORY) 
except OSError as error: 
    print(error) 

# Load the features for the data
x_hinge = np.load("features/hinge_features.npy")
y = np.load("features/labels.npz")['label']

hinge_feature_vectors = []
ecount = 0

start_time = time.time()

img_filenames = os.listdir(TEST_DIRECTORY)
for img_filename in tqdm(img_filenames):
    try:
        img_path = os.path.join(TEST_DIRECTORY,img_filename)
        h_f = get_hinge_features(img_path)
        hinge_feature_vectors.append(h_f)
    except Exception as inst:
        ecount += 1
        if ecount % 20 == 0:
            print(inst, f'error count: {ecount}')
        continue

np.save(os.path.join(OUTPUT_DIRECTORY, f"hinge_features.npy"), hinge_feature_vectors)
print(f"Saved all hinge and cold features")


# Classification
x_hingeT = np.load(f"{OUTPUT_DIRECTORY}/hinge_features.npy")
y_pred_hinge = []

clf = SVC(kernel='rbf', verbose=True, C=10) # Linear Kernel

#Train the model using the training sets
clf.fit(x_hinge, y)

#Predict the response for test dataset

y_pred_hinge = clf.predict(x_hingeT)
diff_time=(time.time() - start_time)


# if os.path.exists("hingeResult.txt"):
#   os.remove("hingeResult.txt")
text_file = open(f"{OUTPUT_DIRECTORY}/results.txt", "w")
for i in y_pred_hinge:
    n = text_file.write(str(f'{i}\n'))
text_file.close()

# if os.path.exists("hingeTime.txt"):
#   os.remove("hingeTime.txt")
text_file = open(f"{OUTPUT_DIRECTORY}/times.txt", "w")
for i in range(0,len(img_filenames)):
    n = text_file.write(str(f'{diff_time/len(img_filenames)}\n'))
text_file.close()
