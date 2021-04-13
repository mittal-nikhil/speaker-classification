import os
import pickle
import numpy as np
from scipy.io.wavfile import read
from feature_extraction import extract_features
import warnings
warnings.filterwarnings("ignore")
import time

source   = "testingData/"   
modelpath = "speakerModels/"

gmm_files = [os.path.join(modelpath,fname) for fname in 
              os.listdir(modelpath) if fname.endswith('.gmm')]

models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname 
              in gmm_files]

error = 0
total_sample = 0.0


print("Do you want to Test a Single Audio: Press '1' or The complete Test Audio Sample: Press '0' ?")
take = int(input().strip())
if take == 1:
        print("Enter the File name from Test Audio Sample Collection :")
        path=input().strip()
        print("Testing Audio : ", path)
        sr,audio=read(source+path)
        vector=extract_features(audio,sr)
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
                gmm    = models[i]
                scores = np.array(gmm.score(vector))
                log_likelihood[i] = scores.sum()
        winner = np.argmax(log_likelihood)
        print("\tdetected as - ", speakers[winner])
        time.sleep(1.0)
elif take == 0:
        test_file = "test_data_file.txt"
        file_paths = open(test_file,'r')
        for path in file_paths:
                total_sample += 1.0
                path = path.strip() 
                print("Testing Audio : ", path)
                sr,audio = read(source + path)
                vector   = extract_features(audio,sr)
                log_likelihood = np.zeros(len(models))
                for i in range(len(models)):
                        gmm    = models[i]
                        scores = np.array(gmm.score(vector))
                        log_likelihood[i] = scores.sum()
                        print(log_likelihood[i],"-",speakers[i])
                winner = np.argmax(log_likelihood)
                print ("\tdetected as - ", speakers[winner])
                checker_name = path.split("/")[0]
                if speakers[winner] != checker_name:
                        error += 1
                time.sleep(1.0)

        print(error, total_sample)
        accuracy = ((total_sample - error) / total_sample) * 100

        print ("The Accuracy Percentage for the current testing Performance with MFCC + GMM is : ", accuracy, "%")

        
