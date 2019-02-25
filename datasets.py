# datasets
import numpy as np
import wget
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_bacteria(train_size=0.8,seed=42):
    url = 'https://www.dropbox.com/s/conkqhwi5pd31yk/bacteria.npz?dl=1'

    filename = wget.download(url)
    data = np.load(filename)
    images = np.expand_dims(data['images']/255,axis=3)
    labels = to_categorical(data['labels'])

    test_size  = 1.-train_size
    X_train, X_test, y_train, y_test = train_test_split(images, labels, train_size = train_size, test_size=test_size, random_state=seed)

    print("Chargement du jeu de données réussi !\n")
    print("Informations sur le jeu de données")
    print("----------------------------------")
    print("- Nombre d'images totales: ", images.shape[0])
    print("- Proportion d'image dans le Training set:", train_size*100,"%")
    print("- Nombre d'images dans le Training set:", X_train.shape[0])
    print("- Nombre d'images dans le Validation set:", X_test.shape[0])
    print("\n")
    print("Forme du tenseur d'entrée (input_shape): ", images.shape[1:])
    print("Nombre de classes de sortie: ", labels.shape[1])
    print("\n")
    print("\n")
