class Normalizer:  
    def __init__(self, form):  
        '''
        yolo: Normalize the pixel values to be between 0 and 1
        efficientnet: Normalize the pixel values to be between -0.5 and 0.5
        minmax: Min-Max scaling to [0,1] range
        zscore: Z-score normalization (standardization)
        imagenet: ImageNet normalization using pre-computed mean and std
        inception: Inception-style normalization to [-1,1]
        resnet: ResNet-style normalization
        '''
        self.form = form  

    def normalize(self, tensor):  
        if self.form == "yolo":  
            """Normalize the pixel values to be between 0 and 1"""  
            tensor = tensor / 255.0  
            
        elif self.form == "efficientnet":  
            """Normalize the pixel values to be between -0.5 and 0.5"""  
            tensor = tensor / 255.0  
            tensor = tensor.sub_(0.5).div_(0.5)  
            
        elif self.form == "minmax":  
            """Min-Max scaling to [0,1] range"""  
            min_val = tensor.min()  
            max_val = tensor.max()  
            tensor = (tensor - min_val) / (max_val - min_val)  
            
        elif self.form == "zscore":  
            """Z-score normalization (standardization)"""  
            mean = tensor.mean()  
            std = tensor.std()  
            tensor = (tensor - mean) / (std + 1e-7)  # 添加小值避免除零  
            
        elif self.form == "imagenet":  
            """ImageNet normalization using pre-computed mean and std"""  
                    # ImageNet 預計算的均值和標準差  
            imagenet_mean = [0.485, 0.456, 0.406]  
            imagenet_std = [0.229, 0.224, 0.225]  
            tensor = tensor / 255.0  # 先歸一化到 [0,1]  
            for t, m, s in zip(tensor, imagenet_mean, imagenet_std):  
                t.sub_(m).div_(s)  
                
        elif self.form == "inception":  
            """Inception-style normalization to [-1,1]"""  
            tensor = ((tensor / 255.0) - 0.5) * 2  
            
        elif self.form == "resnet":  
            """ResNet-style normalization"""  
            tensor = tensor / 255.0  
            tensor[0] = (tensor[0] - 0.485) / 0.229  
            tensor[1] = (tensor[1] - 0.456) / 0.224  
            tensor[2] = (tensor[2] - 0.406) / 0.225  
            
        return tensor  

    def __call__(self, tensor):  
        return self.normalize(tensor)