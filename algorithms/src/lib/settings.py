import os

class Paths:
    IMG_PATH = 'images/'
    DATA_PATH = 'data/'
    
    def ensure_paths_exists():
        if not os.path.exists(Paths.IMG_PATH):
            os.makedirs(Paths.IMG_PATH)
        
        if not os.path.exists(Paths.DATA_PATH):
            os.makedirs(Paths.DATA_PATH)