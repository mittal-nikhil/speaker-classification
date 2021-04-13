import pickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from feature_extraction import extract_features
import warnings
warnings.filterwarnings("ignore")

source_directory="trainingData/"
destination_directory="speakerModels/"
training_file_name="training_data_file.txt"
audio_file_paths=open(training_file_name,'r');

file_count=1
audio_features=np.asarray(())
for audio_file_path in audio_file_paths:
    audio_file_path=audio_file_path.strip()
    print(audio_file_path)
    full_audio_path=source_directory+audio_file_path
    sample_rate,audio=read(full_audio_path)
    vector=extract_features(audio,sample_rate)
    if audio_features.size==0:
        audio_features=vector
    else:
        audio_features=np.vstack((audio_features,vector))
    if file_count==5:
        gmm=GaussianMixture(n_components=16,covariance_type='diag',n_init=3)
        gmm.fit(audio_features)
        pickle_file=audio_file_path.split("/")[0]+".gmm"
        destination_file_path=destination_directory+pickle_file
        pickle.dump(gmm,open(destination_file_path,'wb'))
        print("Model created for speaker: "+pickle_file)
        audio_features=np.asarray(())
        file_count=0
    file_count+=1
        
    
