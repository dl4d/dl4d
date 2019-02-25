#DL4D images object

import pandas as pd
import os
from PIL import Image
import numpy as np

class segmentation:

    def __init__(self):

        self.dataframe = None
        self.X         = None
        self.Y         = None

    def load(self,folder_images,folder_labels,resize,filextension,train_size=0.8,seed=42,normalization=True):
            print('[] Loading images for segmentation ...')
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

            print("[Done]")

    def scaling(self):
        mean = np.mean(self.X);
        std  = np.std(self.X);
        self.X -= mean
        self.X /= std

        if self.Y.max()!=1:
            ma= self.Y.max()
            mi=self.Y.min()
            self.Y = 1-((ma-self.Y)/(ma+mi))
        print("[Done] Data Scaling")


    def infos(self):
        print('\n')
        print("Dataset Informations")
        print("----------------------------------")
        print("- Number of images: ", self.X.shape[0])
        print("- Number of masks : ", self.Y.shape[0])
        if (len(self.X.shape))==4:
            print("- Color channel(s): ", self.X.shape[3])
            print("- Tensor Shape : ", self.X.shape)
        else:
            print("- Color channel(s): ", 1)
            print("- Tensor Shape : ", np.expand_dims(self.X,axis=3).shape)

        print("\n")
