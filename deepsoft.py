import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from sklearn.decomposition import PCA
from sklearn import metrics 
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix
from tabulate import tabulate
from prettytable import PrettyTable
# Reading input 
X = pd.read_csv(r'songs.csv')
# Reading output
Y = pd.read_csv(r'onehotlabels.csv')



# Split into test train sets after shuffling 

idx = np.random.permutation(X.index)
X = X.reindex(idx)
Y = Y.reindex(idx)

X = X.to_numpy()
Y = Y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.25, shuffle = True, random_state = 42)
X_test,X_val,y_test,y_val= train_test_split(X_test, y_test,test_size=0.5, shuffle = True, random_state = 42)


def custom_loss2(train_songs, decoded_songs):
  
    loss = tf.reduce_mean(tf.pow( decoded_songs - train_songs, 2))
    print(loss)
    return loss

inputs = keras.Input(shape = (500,))

def custom_loss1(y_true, y_pred):
  
    loss = tf.reduce_mean(0.1*tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(y_true), logits=y_pred))
    print(loss)
    return loss


x = layers.Dense(256, activation="relu")(inputs)
x = layers.Dense(192, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)

newx1 = layers.Dense(128, activation="sigmoid")(x)
newx1 = layers.Dense(192, activation="sigmoid")(newx1)
newx1 = layers.Dense(256, activation="sigmoid")(newx1)
newx1 = layers.Dense(500, activation="sigmoid", name = "reconstruction")(newx1)


newx2 = layers.Dense(32, activation="tanh")(x)
newx2 = layers.Dense(16, activation="tanh")(newx2)
newx2 = layers.Dense(4 , activation="tanh", name = "classification")(newx2)

model = keras.Model(inputs=inputs, outputs=[newx1,newx2], name="deepsoftmax")


loss1 =  custom_loss1
loss2 = custom_loss2



losses = {
    "reconstruction" : loss1,
    "classification" : loss2,
}
lossWeights = {"reconstruction": 1.0, "classification": 1.0}
model.compile(loss=losses,optimizer="adam",metrics=["accuracy"],loss_weights=lossWeights)

history = model.fit(
    x=X_train, y=[X_train,y_train], 
    epochs= 10, 
    batch_size= 512,
    validation_data=( X_val, [X_val,y_val]), 
)

reconstruction,classification_pred=model.predict(X_test)
# test accuracy
classification_pred_max = np.argmax(classification_pred, axis=1)
y_test_max = np.argmax(y_test, axis=1)
print("Test accuracy: ", metrics.accuracy_score(y_test_max, classification_pred_max))
_, classification_pred_train = model.predict(X_train)
classification_pred_train_max = np.argmax(classification_pred_train, axis=1)
y_train_max = np.argmax(y_train, axis=1)
print("Train accuracy: ", metrics.accuracy_score(y_train_max, classification_pred_train_max))

_, classification_pred_val = model.predict(X_val)
classification_pred_val_max = np.argmax(classification_pred_val, axis=1)
y_val_max = np.argmax(y_val, axis=1)
print("Validation accuracy: ", metrics.accuracy_score(y_val_max, classification_pred_val_max))

pca=PCA(n_components=2)
pca.fit(classification_pred)
pca_classification_pred=pca.transform(classification_pred)



colors=[]

for i in range(len(y_test)):
    if y_test[i][0]==1:
        colors.append('purple')
    elif y_test[i][1]==1:
        colors.append('yellow')
    elif y_test[i][2]==1:
        colors.append('green')
    elif y_test[i][3]==1:
        colors.append('blue')

plt.scatter(pca_classification_pred[:,0],pca_classification_pred[:,1],c=colors,alpha=0.5)

plt.show()











# encoder=Model(inputs=model.input,outputs=model.get_layer('reconstruction').output)
# encoded_test=encoder.predict(X_test)
# pca=PCA(n_components=2)
# pca.fit(encoded_test)
# pca_encoded_test=pca.transform(encoded_test)
# 
# colors=[]
# classical=[]
# jazz=[]
# metal=[]
# pop=[]
# 
# for i in range(len(y_test)):
    # if y_test[i][0]==1:
        # colors.append('blue')
        # classical.append(pca_encoded_test[i])
    # elif y_test[i][1]==1:
        # colors.append('green')
        # jazz.append(pca_encoded_test[i])
    # elif y_test[i][2]==1:
        # colors.append('yellow')
        # metal.append(pca_encoded_test[i])
    # elif y_test[i][3]==1:
        # colors.append('purple')
        # pop.append(pca_encoded_test[i])
# classical=np.array(classical)
# jazz=np.array(jazz)
# metal=np.array(metal)
# pop=np.array(pop)
# alpha=0.4
# plt.scatter(pop[:,0],pop[:,1],c='purple',alpha=alpha)
# 
# plt.scatter(metal[:,0],metal[:,1],c='yellow',alpha=alpha)
# plt.scatter(jazz[:,0],jazz[:,1],c='green',alpha=alpha)
# 
# plt.scatter(classical[:,0],classical[:,1],c='blue',alpha=alpha)
# 
# plt.scatter(pca_encoded_test[:,0],pca_encoded_test[:,1],c=colors,alpha=0.4)
# plt.show()





############################# confusion matrix

classification_pred_max=np.argmax(classification_pred,axis=1)
actual_max=np.argmax(y_test,axis=1)
confusion_matrix=metrics.confusion_matrix(actual_max,classification_pred_max)

confusion_matrix=confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

index=['classical','jazz','metal','pop']
columns=['classical','jazz','metal','pop']
precisions=[]
recalls=[]
f1s=[]
genres=['classical','jazz','metal','pop']
for i in range(0,4):
    
    p=confusion_matrix[i][i]/np.sum(confusion_matrix[i])
    r=confusion_matrix[i][i]/np.sum(confusion_matrix[:,i])
    f1=2*p*r/(p+r)
    precisions.append(p)
    recalls.append(r)
    f1s.append(f1)

# tabulate
# print(tabulate([genres,precisions,recalls,f1s], headers=['genres','precision','recall','f1'], tablefmt='orgtbl'))
table=PrettyTable(['genres','precision','recall','f1'])
for i in range(0,4):
    table.add_row([genres[i],precisions[i],recalls[i],f1s[i]])
print(table)
fig,ax=plot_confusion_matrix(conf_mat=confusion_matrix, show_absolute=False, show_normed=True, colorbar=True,class_names=['classical','jazz','metal','pop'])
plt.show()

precisions=[]
recalls=[]
f1s=[]
genres=['classical','jazz','metal','pop']
for i in range(0,4):
    
    p=confusion_matrix[i][i]/np.sum(confusion_matrix[i])
    r=confusion_matrix[i][i]/np.sum(confusion_matrix[:,i])
    f1=2*p*r/(p+r)
    precisions.append(p)
    recalls.append(r)
    f1s.append(f1)

# tabulate
# print(tabulate([genres,precisions,recalls,f1s], headers=['genres','precision','recall','f1'], tablefmt='orgtbl'))
table=PrettyTable(['genres','precision','recall','f1'])
for i in range(0,4):
    table.add_row([genres[i],precisions[i],recalls[i],f1s[i]])
print(table)