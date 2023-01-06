import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
def custom_loss2(train_songs, decoded_songs):
  
    loss = tf.reduce_mean(tf.pow( decoded_songs - train_songs, 2))
    print(loss)
    return loss
# Load data
songs = pd.read_csv('songs.csv')
one_hot=pd.read_csv('onehotlabels.csv')
index=np.random.permutation(songs.index)
songs=songs.reindex(index)
one_hot=one_hot.reindex(index)

# Split data
train_songs, rem_songs, train_labels, rem_labels = train_test_split(songs, one_hot, test_size=0.25, random_state=42)
val_songs, test_songs, val_labels, test_labels = train_test_split(rem_songs, rem_labels, test_size=0.5, random_state=42)

#encoder
input_song = Input(shape=(500,))
encoded = Dense(64, activation='relu')(input_song)

#decoder
decoded = Dense(500, activation='sigmoid')(encoded)

#autoencoder
autoencoder=Model(inputs=input_song, outputs=decoded)

#compile
autoencoder.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
# train
autoencoder.fit(train_songs, train_songs, epochs=10, batch_size=256, shuffle=True, validation_data=(val_songs, val_songs))

# create encoder model
encoder = Model(input_song, encoded)

# create decoder model
encoded_input = Input(shape=(64,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

# evaluate
loss, acc = autoencoder.evaluate(test_songs, test_songs)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# predict
encoded_songs = encoder.predict(test_songs)
decoded_songs = decoder.predict(encoded_songs)

# PCA visualization

pca = PCA(n_components=2)
pca_result = pca.fit_transform(encoded_songs)

colors=[]
classical=[]
jazz=[]
metal=[]
pop=[]
test_labels=np.array(object=test_labels)
for i in range(len(test_labels)):
    if test_labels[i][0]==1:
        colors.append('blue')
        classical.append(pca_result[i])
    elif test_labels[i][1]==1:
        colors.append('green')
        jazz.append(pca_result[i])
    elif test_labels[i][2]==1:
        colors.append('yellow')
        metal.append(pca_result[i])
    else:
        colors.append('purple')
        pop.append(pca_result[i])

classical=np.array(classical)
jazz=np.array(jazz)
metal=np.array(metal)
pop=np.array(pop)
alpha=0.5
plt.scatter(pca_result[:,0], pca_result[:,1], c=colors,label=colors, alpha=alpha)

# plt.scatter(pop[:,0], pop[:,1],alpha=alpha, c='purple',label='pop')

# plt.scatter(metal[:,0], metal[:,1],alpha=alpha, c='yellow',label='metal')
# plt.scatter(jazz[:,0], jazz[:,1],alpha=alpha, c='green',label='jazz')

# plt.scatter(classical[:,0], classical[:,1],alpha=alpha, c='blue',label='classical')


plt.show()

