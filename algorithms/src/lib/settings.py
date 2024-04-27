import os

class Paths:
    IMG_PATH = 'algorithms/images/'
    
    def ensure_paths_exists():
        if not os.path.exists(Paths.IMG_PATH):
            os.makedirs(Paths.IMG_PATH)
        