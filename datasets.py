# datasets
import numpy as np
import wget
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#Generic load
def load(url,train_size=0.8,seed=42):
    filename = wget.download(url)
    data = np.load(filename)
    images = np.expand_dims(data['images']/255,axis=3)
    labels = to_categorical(data['labels'])

    test_size  = 1.-train_size
    X_train, X_test, y_train, y_test = train_test_split(images, labels, train_size = train_size, test_size=test_size, random_state=seed)

    return (X_train,y_train),(X_test,y_test)

def load_bacteria(train_size=0.8,seed=42):
    print('Loading!!')
    url = 'https://www.dropbox.com/s/conkqhwi5pd31yk/bacteria.npz?dl=1'
    return load(url,train_size,seed)
