from dl4d import *
#from modules.images.classification import classification
from modules.images.segmentation import segmentation


if __name__ == "__main__":

     folder_images = "C:\\Users\\daian\\Desktop\\DATA\\Blob\\images\\"
     folder_labels = "C:\\Users\\daian\\Desktop\\DATA\\Blob\\masks\\"
     resize = (64,64)
     filextension = "jpg"

     s = segmentation()

     s.load(folder_images,folder_labels,resize,filextension)
     s.infos()

     s.scaling()

     from sklearn.model_selection import train_test_split
     X_train,Y_train,X_valid,Y_valid = train_test_split(s.X,s.Y,test_size=0.2,random_state = 42)
