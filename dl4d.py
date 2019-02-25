from modules.images import *

class dl4d:

    def __init__(self,colab=False):

        VERSION = "0.1"
        print('Deep Learning for Dummies (version ' + VERSION + ')')

        # global variables

        #Use google drive as virtual disk
        self.drive = None
        if colab:
            self.gdrive_mount()

        #loader module
        #self.loader    = loader_module()

        #trainer module
        self.trainer   = None

        #monitor module
        self.monitor   = None

        #evaluator module
        self.evaluator = None

    #Mount google drive virtual disk
    def gdrive_mount(self):
        from google.colab import drive
        drive.mount('/content/gdrive')
        self.drive = '/content/gdrive/My Drive/'
