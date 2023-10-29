
#importing necessary libraries
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import keras
from keras import backend as K
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.utils import class_weight
import tensorflow as tf
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.utils import class_weight
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#importing embeddings generated from universal sentence encoder
review_txt=np.load("/kaggle/input/embeddings-use/review_txt.npy")
synopysys=np.load("/kaggle/input/embeddings-use/plot_synopsis.npy")
y=np.load("/kaggle/input/y-label-for-use/y_label.npy")

#doing PCA for visualisation of sepearbility of embeddings
pca = PCA(n_components=2)
X_pca = pca.fit_transform(review_txt[:10000])
# Plot the PCA principal components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y[:10000], cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Review Text')
plt.colorbar(label='Target Class')
plt.show()


#doing TSNE for visualisation of sepearbility of embeddings
tsne = TSNE(n_components=2, random_state=42,perplexity=25.0)
X_tsne = tsne.fit_transform(review_txt[:10000])
# Plot the t-SNE
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y[:10000], cmap='viridis')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Review Text')
plt.colorbar(label='Target Class')
plt.show()

#general architecture of deep 
def deep(optimizer,loss):
    #input embedding is of 512 as Universal Sentence Encoder outputs 512 dimen embeddings
    embedding_input=keras.Input(shape=(512,),name="input")
    embedding_input2=keras.Input(shape=(512,),name="input2")
    #using relu activation functions cause its light weight as compared to other GELU and other things.
    deep = tf.keras.layers.Dense(512, activation='relu',name="Dense_Deep_512")(embedding_input)
    #dropouts to prevent overfitting
    deep=tf.keras.layers.Dropout(.5)(deep)
    deep = tf.keras.layers.Dense(256, activation='relu',name="Dense_Deep_256")(deep)
    #dropouts to prevent overfitting
    deep=tf.keras.layers.Dropout(.5)(deep)
    deep = tf.keras.layers.Dense(128, activation='relu',name="Dense_Deep_128")(deep)
    #batch normalisation to prevent internal covaraiance shift
    deep=tf.keras.layers.BatchNormalization()(deep)
    deep2 = tf.keras.layers.Dense(512, activation='relu',name="Dense_Deep_512_2")(embedding_input2)
    #dropouts to prevent overfitting
    deep2=tf.keras.layers.Dropout(.5)(deep2)
    deep2 = tf.keras.layers.Dense(256, activation='relu',name="Dense_Deep_256_2")(deep2)
    #dropouts to prevent overfitting
    deep2=tf.keras.layers.Dropout(.5)(deep2)
    deep2 = tf.keras.layers.Dense(128, activation='relu',name="Dense_Deep_128_2")(deep2)
    #batch normalisation to prevent internal covaraiance shift
    deep2=tf.keras.layers.BatchNormalization()(deep2)
    #dot prouduct of emmbeddings instead of concatination cause dot product measures similarity
    both=tf.keras.layers.Dot(axes=1)([deep, deep2])
    #followed by dense layers
    both = tf.keras.layers.Dense(128, activation='relu',name="FF_2_128")(both)
    both = tf.keras.layers.Dense(128, activation='relu',name="FF_2_1288")(both)
    #sigmoid for binary classification
    output = tf.keras.layers.Dense(1, activation='sigmoid',name='sigmoid')(both)
    keras_model = tf.keras.models.Model([embedding_input,embedding_input2], output)
    keras_model.compile(optimizer=optimizer,loss=loss,metrics=[loss,tf.keras.metrics.TruePositives()])
    return keras_model

#plotting function
tf.keras.utils.plot_model(deep(tf.keras.optimizers.legacy.Adam(learning_rate=0.001),"binary_crossentropy"), "my_first_model.png",show_shapes=True)


#train test split of 80-20 raion
X_train, X_test, y_train, y_test = train_test_split(np.concatenate((review_txt,synopysys),axis=1), y, test_size=0.2, random_state=42)
#concatinating & de concatinating of embeddings
X_train_left,X_train_right=np.split(X_train,2,axis=1)
X_test_left,X_test_right=np.split(X_test,2,axis=1)

#this code calculates class weights as there is class imbalance
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(y_train),
                                                 y=y_train.ravel())


#printing summary of model
deep(tf.keras.optimizers.legacy.Adam(learning_rate=0.001),"binary_crossentropy").summary()

#early stopping based on val_loss & warmup period of 40
early_stopping=tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=5,verbose=0,mode="auto",baseline=None,restore_best_weights=True,start_from_epoch=40,)


def save_model(i):
    model_dic[i].save('path/to/save/model_{}.h5'.format(i))
    model_json = model_dic[i].to_json()
    with open('path/to/save/model_{}.json'.format(i), 'w') as json_file:
        json_file.write(model_json)
    model_dic[i].save_weights('path/to/save/model_weights{}.h5'.format(i))



# list of batch sizes to experiment with 
batch_sizes = [128, 256, 512]
history_per_batch_size = {}
model_dic={}

# Iterate through each batch size and train the model
for batch_size in batch_sizes:
    # Define the model and compile it
    model = deep(tf.keras.optimizers.legacy.Adam(learning_rate=0.001),"binary_crossentropy")  # Assume create_model() function is defined as before
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model with the current batch size
    history = model.fit(
        x={"input": X_train_left, 'input2': X_train_right},
        y=y_train,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=250,
        class_weight={0: class_weights[0], 1: class_weights[1]},
        callbacks=[early_stopping]
    )
    
    # Store the training history for this batch size
    history_per_batch_size[batch_size] = history
    model_dic[batch_size]=model
#     save_model(batch_size)
    
    

#plotting the validation loss for checking overfitting
plt.plot(history_per_batch_size[128].history['loss'],label='Training Loss (Batch Size 128)',color='blue')
plt.plot(history_per_batch_size[128].history['val_loss'],label='Validation Loss (Batch Size 128)',linestyle='-.',color='blue')
plt.plot(history_per_batch_size[256].history['loss'],label='Training Loss (Batch Size 256)',color='red')
plt.plot(history_per_batch_size[256].history['val_loss'],label='Validation Loss (Batch Size 256)',linestyle='-.',color='red')
plt.plot(history_per_batch_size[512].history['loss'],label='Training Loss (Batch Size 512)',color='green')
plt.plot(history_per_batch_size[512 ].history['val_loss'],label='Validation Loss (Batch Size 512)',linestyle='-.',color='green')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


#testing our 128 batch size model
y_pred_128=model_dic[128].predict(x={"input": X_test_left,'input2': X_test_right})
preds_128=np.where(y_pred_128> 0.50, 1, 0)

#testing our 256 batch size model
y_pred_256=model_dic[256].predict(x={"input": X_test_left,'input2': X_test_right})
preds_256=np.where(y_pred_256> 0.50, 1, 0)

#testing our 512 batch size model
y_pred_512=model_dic[512].predict(x={"input": X_test_left,'input2': X_test_right})
preds_512=np.where(y_pred_512> 0.50, 1, 0)


#ploting confusion matrix of 128 batch size model
cm=confusion_matrix(y_test, preds_128)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("128 batch size")
plt.show()


#ploting confusion matrix of 256 batch size model
cm=confusion_matrix(y_test, preds_256)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("256 batch size")
plt.show()

#ploting confusion matrix of 512 batch size model
cm=confusion_matrix(y_test, preds_512)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("512 batch size")
plt.show()

#ploting ROC curve of 128 batch size model
RocCurveDisplay.from_predictions(
    y_test,
    y_pred_128
)

#ploting ROC curve of 256 batch size model
RocCurveDisplay.from_predictions(
    y_test,
    y_pred_256
)

#ploting ROC curve of 512 batch size model
RocCurveDisplay.from_predictions(
    y_test,
    y_pred_512
)


#printing classification report of 128 batch size model
report = classification_report(y_test, preds_128)
print("Classification Report:\n", report)


#printing classification report of 256 batch size model
report = classification_report(y_test, preds_256)
print("Classification Report:\n", report)




#printing classification report of 512 batch size model
report = classification_report(y_test, preds_512)
print("Classification Report:\n", report)


#saving weights of 3 models with batch size 128,256,512
for i in [128,256,512]:
    model_dic[i].save_weights(os.path.join("{}_batch_size_contrastive_model".format(i), 'model_weights_contrastive_model_{}.h5'.format(i)))
