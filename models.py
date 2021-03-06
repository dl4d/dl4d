from keras.utils import plot_model
import matplotlib.pyplot as plt

from keras.optimizers import Adam
import keras.backend as K

import pandas as pd


class model:

    def __init__(self,name="NoName"):

        model   = None
        history = None
        name    = name
        #K.clear_session()


    def assign(self,model):
        self.model=model


    def train(self,data,validation_data):

        self.model.compile(optimizer=Adam(lr=1e-3),loss="categorical_crossentropy",metrics=["accuracy"])

        history = self.model.fit(data[0],data[1],validation_data=(validation_data[0],validation_data[1]),epochs=10)

        if hasattr(self, 'history'):
            self.history.history[self.model.metrics_names[0]]+=history.history[self.model.metrics_names[0]]
            self.history.history["val_"+self.model.metrics_names[0]]+=history.history["val_"+self.model.metrics_names[0]]
            self.history.history[self.model.metrics_names[1]]+=history.history[self.model.metrics_names[1]]
            self.history.history["val_"+self.model.metrics_names[1]]+=history.history["val_"+self.model.metrics_names[1]]
        else:
            self.history = history

    #    if self.history is None:
    #        self.history = history
    #    else:


    def plot_model_tree(self):
        plot_model(self.model, to_file='tmp_model.png')
        plt.figure(figsize=(20,20))
        plt.imshow(plt.imread("tmp_model.png"))
        plt.axis('off')
        plt.show()

    def confusion_matrix(self,test):
        p = self.model.predict(test[0])
        return pd.crosstab(
                pd.Series(test[1].argmax(axis=1), name='Validation'),
                pd.Series(p.argmax(axis=1), name='Prediction')
                )

    def show_learning_graph(self):
        #Training / Validation graph
        # summarize history for accuracy
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title(self.name +' Performance')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title(self.name+' Error')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()
