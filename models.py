from keras.utils import plot_model
import matplolib.pyplot as plt

def plot_model(model):
    plot_model(model, to_file='tmp_model.png')
    plt.figure(figsize=(20,20))
    plt.imshow(plt.imread("tmp_model.png"))
    plt.axis('off')
    plt.show()
