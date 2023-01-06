import tensorflow as tf
from tensorflow import keras
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
import keras.backend as K
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model
from pandas.plotting import table
def custom_loss(y_true, y_pred):

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.stop_gradient(y_true), logits=y_pred))
    print(loss)
    return loss
# Reading input
songs=pd.read_csv('songs.csv')
labels=pd.read_csv('labels.csv')
one_hot=pd.read_csv('onehotlabels.csv')
#shuffle
index=np.random.permutation(songs.index)
songs=songs.reindex(index)
labels=labels.reindex(index)
one_hot=one_hot.reindex(index)

songs=songs.to_numpy()
labels=labels.to_numpy()
one_hot=one_hot.to_numpy()

# Split data
train_songs, rem_songs, train_labels, rem_labels = train_test_split(songs, one_hot, test_size=0.25,shuffle=True, random_state=42)
val_songs, test_songs, val_labels, test_labels = train_test_split(rem_songs, rem_labels, test_size=0.5,shuffle=True, random_state=42)

# Create model
input_song = keras.Input(shape=(500,))
hidden = Dense(128, activation='tanh')(input_song)
output = Dense(4)(hidden)

# Compile model
two_layer=Model(inputs=input_song,outputs=output)
two_layer.compile(optimizer='adam',loss=custom_loss,metrics=['accuracy'])

# Train model
two_layer.fit(train_songs,train_labels,epochs=8,batch_size=512,validation_data=(val_songs,val_labels))

# Evaluate model
two_layer.evaluate(test_songs,test_labels)
#

test_pred=two_layer.predict(test_songs)
test_pred_max=np.argmax(test_pred,axis=1)
test_max=np.argmax(test_labels,axis=1)

# Calculate accuracy
test_accuracy=np.sum(test_pred_max==test_max)/len(test_max)
print("Test accuracy:",test_accuracy)

train_pred=two_layer.predict(train_songs)
train_pred_max=np.argmax(train_pred,axis=1)
train_max=np.argmax(train_labels,axis=1)

# Calculate accuracy
train_accuracy=np.sum(train_pred_max==train_max)/len(train_max)
print("Train accuracy:",train_accuracy)

val_pred=two_layer.predict(val_songs)
val_pred_max=np.argmax(val_pred,axis=1)
val_max=np.argmax(val_labels,axis=1)

# Calculate accuracy
val_accuracy=np.sum(val_pred_max==val_max)/len(val_max)
print("Validation accuracy:",val_accuracy)




######################3
# PCA visualization

pca = PCA(n_components=2)
pca.fit(test_pred)

test_pred_pca=pca.transform(test_pred)


colors = []
classical = []
jazz = []
metal = []
pop = []
# df=pd.DataFrame(columns=['color','x','y'])

temp_list=[]

for i in range(len(test_pred_pca)):
    if test_max[i]==0:
        colors.append('blue')
        temp_list.append(['blue',test_pred_pca[i][0],test_pred_pca[i][1]])
        classical.append(test_pred_pca[i])
    elif test_max[i]==1:
        colors.append('green')
        jazz.append(test_pred_pca[i])
        temp_list.append(['green',test_pred_pca[i][0],test_pred_pca[i][1]])
    elif test_max[i]==2:
        colors.append('yellow')
        metal.append(test_pred_pca[i])
        temp_list.append(['yellow',test_pred_pca[i][0],test_pred_pca[i][1]])
    else:
        colors.append('purple')
        pop.append(test_pred_pca[i])
        temp_list.append(['purple',test_pred_pca[i][0],test_pred_pca[i][1]])
classical=np.array(classical)
jazz=np.array(jazz)
metal=np.array(metal)
pop=np.array(pop)
alpha=0.5
# df=pd.DataFrame(temp_list,columns=['color','x','y'])
plt.scatter(pop[:, 0], pop[:, 1], c='purple', label='pop', alpha=alpha)
plt.scatter(metal[:, 0], metal[:, 1], c='yellow', label='metal', alpha=alpha)
plt.scatter(jazz[:, 0], jazz[:, 1], c='green', label='jazz', alpha=alpha)
plt.scatter(classical[:, 0], classical[:, 1],c='blue', label='classical', alpha=alpha)

# plt.scatter(test_pred_pca[:,1],test_pred_pca[:,0],c=colors,alpha=alpha)


list_colors=['blue','green','yellow','purple']
list_labels=['classical','jazz','metal','pop']
plt.title("PCA of 4-dim output of 2-layer NN")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
# table(loc='upper right',cellText=list_labels,colWidths=[0.1]*len(list_labels),cellColours=list_colors)
# plt.table(cellText=list_labels,cellColours=list_colors,loc='upper right')
plt.show()
