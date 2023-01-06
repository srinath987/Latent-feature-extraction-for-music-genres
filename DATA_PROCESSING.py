import os
# import sox
import numpy as np
import librosa
import pandas as pd
import numpy as np

def get_one_hot(labels, num_classes=4):
    return np.eye(num_classes)[labels]

songs=np.zeros((12000,500))
onehotlabels=np.zeros((12000,4))
counter=0
allgenres=['classical','jazz','metal','pop']
numsplit=30
sizesplit=500
labels=np.zeros(12000)
for index in range(len(allgenres)):
    for filename in os.listdir('./wav/Data/genres_original/' + allgenres[index]):

        print(filename)
        if filename.endswith(".wav"):
            y, sr = librosa.load('./wav/Data/genres_original/' + allgenres[index]+'/'+filename)
            # testmfcc = librosa.feature.mfcc(y=y, sr=sr)
            print(y.shape)
            y=y[0:600000]
            y=y.reshape(15000,40)
            y=np.mean(y,axis=1)
            for i in range(numsplit):
                songs[counter]=y[i*sizesplit:(i+1)*sizesplit]
                onehotlabels[counter] = get_one_hot(index)
                labels[counter]=index
                counter+=1
print(songs.shape)
print(songs)

songs = pd.DataFrame(songs)
labels = pd.DataFrame(labels)
onehotlabels = pd.DataFrame(onehotlabels)
songs.to_csv('songs.csv', index = False)
labels.to_csv('labels.csv', index = False)
onehotlabels.to_csv('onehotlabels.csv', index = False)
print('Conversion done')
 