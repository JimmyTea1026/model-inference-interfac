
class Normalizer:
    def __init__(self, form):
        self.form = form

    def normalize(self, tensor):
        if self.form == "yolo":
            """Normalize the pixel values to be between -1 and 1"""
            tensor = tensor / 255.0
        elif self.form == "efficientnet":
            """Normalize the pixel values to be between -0.5 and 0.5"""
            tensor = tensor / 255.0
            tensor = tensor.sub_(0.5).div_(0.5)

    def __call__(self, tensor):
        return self.normalize(tensor)