import random
import matplotlib.pyplot as plt
import numpy as np

def show_random_images(X,y,label=None,show_mask=False):
    num=[x for x in range(0,X.shape[0])]
    random.shuffle(num)
    fig = plt.figure(figsize=(10,10))

    if label is not None:
        for i in range(0,5):
            plt.subplot(1,5,i+1)
            #Showing denormalized images
            tmp = X[np.where(y.argmax(axis=1)==label)]
            num=[x for x in range(0,tmp.shape[0])]
            random.shuffle(num)
            plt.imshow(np.squeeze(tmp[num[i]]),cmap='gray')
            plt.axis('off')
            plt.title(str(label))
