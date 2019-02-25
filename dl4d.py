import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import RMSprop,SGD,Adam,Adamax,Nadam,Adadelta,Adagrad

from keras.applications.mobilenet_v2 import preprocess_input


from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import random
from random import randint

import requests
from io import BytesIO
import pickle
from urllib.request import urlopen




class formation:

    def __init__(self,mount=False):
        X_train = None
        X_valid = None
        y_train = None
        y_valid = None
        X       = None
        Y       = None
        synsets = None
        data    = None

        model = None
        history = None

        if mount:
            self.gdrive_mount()


    def load_bacteria_dataset_features(self,train_size=0.8,seed=42,normalization=True,overfit=True):
        data = pd.read_csv('./bacteria.csv')
        Y= to_categorical(data.Label)
        data = data.drop(columns=["Id","Label"])
        if overfit:
            data = data.drop(columns=["Area","Mean","StdDev","Mode","Min","Max","X","Y","Feret","FeretX","Slice"])
        self.data = data
        X = data.values
        if normalization==True:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        self.X = X
        self.Y = Y
        test_size  = 1.-train_size
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, Y, train_size = train_size, test_size=test_size, random_state=seed)

        print("Chargement du jeu de données réussi !\n")
        print("Informations sur le jeu de données")
        print("----------------------------------")
        print("- Nombre d'individus total: ", self.X.shape[0])
        print("- Nombre de features totale: ", self.X.shape[1])
        print("- Proportion d'individus dans le Training set:", train_size*100,"%")
        print("- Nombre d'individus dans le Training set:", self.X_train.shape[0])
        print("- Nombre d'individus dans le Validation set:", self.X_valid.shape[0])
        print("\n")
        print("Forme du tenseur d'entrée (input_shape): ", self.X.shape[1:])
        print("Nombre de classes de sortie: ", self.Y.shape[1])
        print("\n")
        if normalization==True:
            print("Normalisation: OUI : Zscore")
        else:
            print("Normalisation: NON")
        print("\n")




    def load_bacteria_dataset_images(self,train_size=0.8,seed=42):
        #Load the file
        data = np.load('./bacteria.npz')
        images = np.expand_dims(data['images']/255,axis=3)
        labels = to_categorical(data['labels'])

        test_size  = 1.-train_size
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(images, labels, train_size = train_size, test_size=test_size, random_state=seed)

        self.X=images
        self.Y=labels

        print("Chargement du jeu de données réussi !\n")
        print("Informations sur le jeu de données")
        print("----------------------------------")
        print("- Nombre d'images totales: ", images.shape[0])
        print("- Proportion d'image dans le Training set:", train_size*100,"%")
        print("- Nombre d'images dans le Training set:", self.X_train.shape[0])
        print("- Nombre d'images dans le Validation set:", self.X_valid.shape[0])
        print("\n")
        print("Forme du tenseur d'entrée (input_shape): ", images.shape[1:])
        print("Nombre de classes de sortie: ", labels.shape[1])
        print("\n")
        print("\n")
        #return X_train, X_valid, y_train, y_valid


    def load_mnist_dataset(self):
        from keras.datasets import mnist
        from keras import backend as K

        (self.X_train, self.y_train), (self.X_valid, self.y_valid) = mnist.load_data()


        self.X_train = self.X_train.reshape(self.X_train.shape[0], 28,28, 1)
        self.X_valid = self.X_valid.reshape(self.X_valid.shape[0], 28,28, 1)

        self.X_train = self.X_train.astype('float32')
        self.X_valid = self.X_valid.astype('float32')
        self.X_train /= 255
        self.X_valid /= 255

        self.y_train = keras.utils.to_categorical(self.y_train, 10)
        self.y_valid = keras.utils.to_categorical(self.y_valid, 10)

        print("Chargement du jeu de données réussi !\n")
        print("Informations sur le jeu de données")
        print("----------------------------------")
        print("- Nombre d'images totales: ", (self.X_train.shape[0] + self.X_valid.shape[0]))
        pptrain = np.round(self.X_train.shape[0]/(self.X_train.shape[0]+self.X_valid.shape[0])*100)
        print("- Proportion d'image dans le Training set:", pptrain,"%")
        print("- Nombre d'images dans le Training set:", self.X_train.shape[0])
        print("- Nombre d'images dans le Validation set:", self.X_valid.shape[0])
        print("\n")
        print("Forme du tenseur d'entrée (input_shape): ", self.X_train.shape[1:])
        print("Nombre de classes de sortie: ", self.y_train.shape[0])
        print("\n")
        print("\n")



    def load_skin_dataset(self,train_size=0.8,seed=42,preprocess=None):

        data    = np.load('./skin.npz')

        images = data['images'].astype("float32")

        labels  = to_categorical(data['labels'])
        self.synsets = data['synsets']

        if preprocess == None:
            images = data['images']/255
        elif preprocess == "InceptionV3":
            images = images[:,:,:,::-1] #RGB -> BGR
            images[:,:,:,0] -= 103.939
            images[:,:,:,1] -= 116.779
            images[:,:,:,2] -= 123.68
        elif preprocess =="MobileNet":
            images = preprocess_input(images)

        else:
            print('Mauvais nom pour le preprocessing input: La liste valide est "InceptionV3" ou rien du tout !')
            return None


        test_size  = 1.-train_size
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(images, labels, train_size = train_size, test_size=test_size, random_state=seed)

        print("Chargement du jeu de données réussi !\n")
        print("Informations sur le jeu de données")
        print("----------------------------------")
        print("- Nombre d'images totales: ", images.shape[0])
        print("- Proportion d'image dans le Training set:", train_size*100,"%")
        print("- Nombre d'images dans le Training set:", self.X_train.shape[0])
        print("- Nombre d'images dans le Validation set:", self.X_valid.shape[0])
        print("\n")
        print("Forme du tenseur d'entrée (input_shape): ", images.shape[1:])
        print("Nombre de classes de sortie: ", labels.shape[1])





    def assign_model(self,model):
        self.model = model


    def train_model(self,model,epochs=100,lr=0.1,learning_rate=None,decay=0.0, optimizer_name='SGD',loss='categorical_crossentropy',metric=["accuracy"],batch_size=20,verbose=2,data_augmentation=None,callbacks=None):

        if learning_rate is not None:
            lr = learning_rate

        self.assign_model(model)

        if optimizer_name=="RMSprop":
            optimizer = RMSprop(lr=lr,decay=decay)
        elif optimizer_name=="SGD":
            optimizer = SGD(lr=lr,decay=decay)
        elif optimizer_name=="Adam":
            optimizer = Adam(lr=lr,decay=decay)
        elif optimizer_name=="Adadelta":
            optimizer = Adadelta(lr=lr,decay=decay)
        else:
            print('Mauvais nom pour l optimizer ! La liste valide est RMSprop,SGD ou Adam !')
            return None

        loss = loss
        metric = metric

        self.model.compile(optimizer,loss,metric)

        if data_augmentation==None:
            self.history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_valid,self.y_valid),
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks
            )
        else:
            self.history=self.model.fit_generator(
                data_augmentation.flow(self.X_train, self.y_train, batch_size=batch_size),
                epochs=epochs,
                steps_per_epoch=self.X_train.shape[0] // batch_size,
                verbose=verbose,
                validation_data=(self.X_valid, self.y_valid),
                callbacks=callbacks
                )






    def show_learning_graph(self):
        #Training / Validation graph
        # summarize history for accuracy
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Performance du modèle')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Erreur du modèle')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()


    def show_features(self,howmany=10,show_normalized=False):
        print("Showing ",howmany," features.\n")
        if show_normalized==False:
            #data = pd.read_csv('./bacteria.csv')
            data = self.data
            return data.head(howmany)
        else:
            print(self.X[0:(howmany-1),:])

    def show_random_images(self,label=None,show_mask=False):
        num=[x for x in range(0,self.X_train.shape[0])]
        random.shuffle(num)
        fig = plt.figure(figsize=(10,10))

        if label is not None:
            for i in range(0,5):
                plt.subplot(1,5,i+1)
                #Showing denormalized images
                tmp = self.X_train[np.where(self.y_train.argmax(axis=1)==label)]
                num=[x for x in range(0,tmp.shape[0])]
                random.shuffle(num)
                plt.imshow(np.squeeze(tmp[num[i]]),cmap='gray')
                plt.axis('off')

        else:
            if show_mask==False:
                for i in range(0,20):
                    plt.subplot(4, 5, i+1)
                    a = self.X_train[num[i]]
                    b = (a - np.min(a))/np.ptp(a)
                    plt.imshow(np.squeeze(b),cmap='gray')
                    lab = np.argmax(self.y_train[num[i]])
                    plt.axis('off')
            else:
                for i in range (0,20):
                    plt.subplot(4,5,i+1)
                    a = self.X_train[num[i]]
                    b = (a - np.min(a))/np.ptp(a)
                    plt.imshow(np.squeeze(b),cmap='gray')
                    plt.axis('off')
                fig = plt.figure(figsize=(10,10))
                for i in range(0,20):
                    plt.subplot(4,5,i+1)
                    a = self.y_train[num[i]]
                    b = (a - np.min(a))/np.ptp(a)
                    plt.imshow(np.squeeze(b),cmap='gray')
                    plt.axis('off')




    def confusion_matrix_model(self):
        p = self.model.predict(self.X_valid)
        return pd.crosstab(
                pd.Series(self.y_valid.argmax(axis=1), name='Validation'),
                pd.Series(p.argmax(axis=1), name='Prediction')
                )

    #ONE Folder containing one subfolder per class
    def load_images_classification(self,folder,resize,filextension):
        images  = []
        labels  = []
        synsets = []
        img_rows=resize[0]
        img_cols=resize[1]

        print(folder)

        k=0
        for d in os.listdir(folder):
            if os.path.isdir(os.path.join(folder, d)):
                curdir = os.path.join(folder,d)
                for filename in os.listdir(curdir):
                    curimg = os.path.join(curdir, filename)
                    if curimg.endswith(filextension):
                        img = Image.open(curimg)
                        resize = img.resize((img_rows,img_cols), Image.NEAREST)
                        images.append(resize)
                        labels.append(k)
                synsets.append(d)
                k=k+1
        imgarray=list();
        for i in range(len(images)):
            tmp = np.array(images[i])
            imgarray.append(tmp)
        imgarray = np.asarray(imgarray)

        synsets=dict(enumerate(np.unique(synsets)))


        self.X = imgarray.astype('float32')
        self.Y = labels
        self.synsets = synsets




    #TWO folder, one for the images, one for the masks
    def load_images_regression(self,folder_images,folder_labels,resize,filextension,train_size=0.8,seed=42,normalization=True):
        images  = []
        labels  = []
        img_rows=resize[0]
        img_cols=resize[1]
        for filename in sorted(os.listdir(folder_images)):
            curimg = os.path.join(folder_images, filename)
            if curimg.endswith(filextension):
                img = Image.open(curimg)
                resize = img.resize((img_rows,img_cols), Image.NEAREST)
                images.append(resize)
        imgarray=list();
        for i in range(len(images)):
            tmp = np.array(images[i])
            imgarray.append(tmp)
        imgarray = np.asarray(imgarray)


        for filename in sorted(os.listdir(folder_labels)):
            curimg = os.path.join(folder_labels, filename)
            if curimg.endswith(filextension):
                img = Image.open(curimg)
                resize = img.resize((img_rows,img_cols), Image.NEAREST)
                labels.append(resize)
        labarray=list();
        for i in range(len(labels)):
            tmp = np.array(labels[i])
            labarray.append(tmp)
        labarray = np.asarray(labarray)

        self.X = imgarray.astype('float32')
        self.Y = labarray.astype('float32')


        if normalization==True:
            mean = np.mean(self.X);
            std  = np.std(self.X);
            self.X -= mean
            self.X /= std

            if self.Y.max()!=1:
                ma= self.Y.max()
                mi=self.Y.min()
                self.Y = 1-((ma-self.Y)/(ma+mi))


        test_size  = 1.-train_size
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X, self.Y, train_size = train_size, test_size=test_size, random_state=seed)


        if len(self.X_train.shape)==3:
            self.X_train = np.expand_dims(self.X_train,axis=3)
            self.X_valid = np.expand_dims(self.X_valid,axis=3)
        if len(self.y_train.shape)==3:
            self.y_train = np.expand_dims(self.y_train,axis=3)
            self.y_valid = np.expand_dims(self.y_valid,axis=3)


        print("Chargement du jeu de données réussi !\n")
        print("Informations sur le jeu de données")
        print("----------------------------------")
        print("- Nombre d'images totales: ", self.X.shape[0])
        print("- Nombre de masques totals: ", self.Y.shape[0])
        print("- Proportion d'image dans le Training set:", train_size*100,"%")
        print("- Nombre d'images dans le Training set:", self.X_train.shape[0])
        print("- Nombre de masques dans le Training set:", self.y_train.shape[0])
        print("- Nombre d'images dans le Validation set:", self.X_valid.shape[0])
        print("- Nombre de masques dans le Validation set:", self.y_valid.shape[0])
        print("\n")
        print("Forme du tenseur d'entrée (input_shape): ", self.X_train.shape[1:])
        print("\n")
        if normalization==True:
            print("- Pixel range des images après normalization:  [",self.X.min(),';',self.X.max(),']')
            print("- Pixel range des masques après normalization: [",self.Y.min(),';',self.Y.max(),']')
        else:
            print("- Pixel range des images:  [",self.X.min(),';',self.X.max(),']')
            print("- Pixel range des masques: [",self.Y.min(),';',self.Y.max(),']')

        print("\n")

    def normalize_images(self,type="StandardScaler",labels=True):
        if type == "StandardScaler":
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)
            if np.array_equal(self.X.shape,self.Y.shape) and label==True:
                self.Y = scaler.fit_transform(self.Y)
        elif type == "MinMaxScaler":
            scaler = MinMaxScaler()
            self.X = scaler.fit_transform(self.X)
            if np.array_equal(self.X.shape,self.Y.shape) and labels==True:
                self.Y = scaler.fit_transform(self.Y)
        elif type == "255":
            self.X = self.X/255
            if np.array_equal(self.X.shape,self.Y.shape) and labels==True:
                self.Y = self.Y/255
        else:
            print('Autorisé: StandardScaler, MinMaxScaler ou 255')


    def train_test_split(self,train_size=0.8,seed=42):
        if self.X is None:
            print('Erreur: self.X vide !')
            return
        test_size  = 1.-train_size
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X, self.Y, train_size = train_size, test_size=test_size, random_state=seed)
        self.X_train = np.expand_dims(self.X_train,axis=3)
        self.X_valid = np.expand_dims(self.X_valid,axis=3)
        self.y_train = np.expand_dims(self.y_train,axis=3)
        self.y_valid = np.expand_dims(self.y_valid,axis=3)


    def gdrive_mount(self):
        from google.colab import drive
        drive.mount('/content/gdrive')
        self.drive = '/content/gdrive/My Drive/'




    def show_activations(self,to_layer,data,figsize=(10,10),random=False,indix=0):
        functor = K.function([self.model.layers[0].input],[self.model.layers[to_layer].output])
        output = functor([data])[0]
        num_filter = K.int_shape(self.model.layers[to_layer].output)[3]
        fig = plt.figure(figsize=(2,2))
        fig.suptitle('Original Image')
        if random==True:
            image_indix = randint(0,output.shape[0])
        else:
            image_indix = indix
        if data.shape[3]==1:
            plt.imshow(np.squeeze(data[image_indix,:,:,:]),cmap='gray')
        else:
            plt.imshow(np.squeeze(data[image_indix,:,:,:]))
        fig = plt.figure(figsize=(round(np.sqrt(num_filter)+1)*2,round(np.sqrt(num_filter)+1)*2))
        fig.suptitle(self.model.layers[to_layer].name)
        for i in range(0,num_filter):
            plt.subplot(round(np.sqrt(num_filter)+1), round(np.sqrt(num_filter)) , i+1)
            plt.title("Filter: "+str(i))
            if (data.shape[3]==1):
                plt.imshow(output[image_indix,:,:,i],cmap="gray")
            else:
                plt.imshow(output[image_indix,:,:,i])
            plt.axis('off')

    def show_neurons(self,layer):
        w = self.model.layers[layer].get_weights()[0]
        f0 = w.shape[2]
        num_filter = w.shape[3]
        fig = plt.figure(figsize=(num_filter,f0+1))
        fig.suptitle(self.model.layers[layer].name)
        k=0
        for j in range(0,f0):
            for i in range(0,num_filter):
                plt.subplot(f0, num_filter , k+1)
                plt.rcParams["axes.grid"] = False
                plt.title("Filter: "+str(i))
                plt.imshow(w[:,:,j,i],cmap="gray")
                plt.axis('off')
                k=k+1

    def show_segmentation(self,N_IMAGE=0,bin_threshold=0.5):
        # - On passe le modèle sur l'image selectionné et on récupère la prédiction
        p = self.model.predict(np.expand_dims(self.X_valid[N_IMAGE],axis=0))

        # - On affiche le résultat
        fig = plt.figure(figsize=(30,30))

        plt.subplot(1,5,1)
        plt.imshow(np.squeeze(self.X_valid[N_IMAGE]))
        plt.axis('off')
        plt.title('Original Image')

        plt.subplot(1,5,2)
        plt.imshow(np.squeeze(self.y_valid[N_IMAGE]))
        plt.axis('off')
        plt.title('Original Mask')

        plt.subplot(1,5,3)
        plt.imshow(np.squeeze(p)>bin_threshold,cmap="gray")
        plt.axis('off')
        plt.title('Learned Mask')

        plt.subplot(1,5,4)
        plt.imshow(np.squeeze(p - self.y_valid[N_IMAGE]))
        plt.axis('off')
        plt.title('Error')

        plt.subplot(1,5,5)
        plt.imshow((np.squeeze(p)>bin_threshold)*np.squeeze(self.X_valid[N_IMAGE]))
        plt.axis('off')
        plt.title('Segmentation')



    def predict_image_from_url(self,model, url,img_size=(128,128), preprocessing="MobileNet",top=5):
        #Pickle synset
        synset = pickle.load(urlopen('https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl') )

        #Get image from url
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

        #Resize the image
        resize = img.resize(img_size, Image.NEAREST)
        resize = np.asarray(resize)
        plt.imshow(resize)
        plt.axis('off')
        resize = resize.astype('float32')
        resize = np.expand_dims(resize,axis=0)

        #Preprocessing
        if preprocessing == "MobileNet":
            resize = preprocess_input(resize)
        #Prediction
        p = model.predict(resize)
        ind_five = (-p).argsort()[-3:][::-1][0][0:top]
        for i in range(top):
            print(synset[ind_five[i]]," : ", np.around(p[0][ind_five[i]]*100,3),"%")


    def predict_from_image(self,model,N_IMAGE):
        a = self.X_valid[N_IMAGE]
        b = (a - np.min(a))/np.ptp(a)
        plt.imshow(np.squeeze(b),cmap='gray')
        plt.axis('off')
        p=model.predict(np.expand_dims(b,axis=0))
        print("Classe prédite: ",np.argmax(p))
        print("Classe réelle : ",np.argmax(self.y_valid[N_IMAGE]))

    def predict_from_features(self,model,N_FEAT):
        p = model.predict(np.expand_dims(self.X_valid[N_FEAT],axis=0))
        print("Classe prédite: ",np.argmax(p))
        print("Classe réelle : ",np.argmax(self.y_valid[N_FEAT]))
