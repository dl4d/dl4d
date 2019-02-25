from keras.utils import plot_model
import matplotlib.pyplot as plt

from keras.optimizers import Adam


def plot_model_tree(model):
    plot_model(model, to_file='tmp_model.png')
    plt.figure(figsize=(20,20))
    plt.imshow(plt.imread("tmp_model.png"))
    plt.axis('off')
    plt.show()

def train(data,validation_data,model):

    model.compile(optimizer=Adam(lr=1e-3),loss="categorical_crossentropy",metrics=["accuracy"])

    model.fit(data[0],data[1],validation_data=(validation_data[0],validation_data[1]),epochs=10)

    return model
