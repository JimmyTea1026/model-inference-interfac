import torch  
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from BaseModel import BaseModel

class YOLOModel(BaseModel):      
    def __init__(self, height: int = 640, width: int = 640, conf_threshold: float = 0.2):  
        super().__init__()
        self.__model_height = height
        self.__model_width = width
        self.__conf_threshold = conf_threshold
    
    def _load_model_online(self):
        # 從網路載入預訓練模型，載入yolov10模型
        self._model = YOLO("yolov10n.pt")
        return 
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:  
        # 實作 YOLO 的預處理邏輯  
        image = cv2.resize(image, (self.__model_width, self.__model_height))
        tensor = torch.from_numpy(image).float()  
        tensor = tensor.permute(2, 0, 1)  # HWC to CHW  
        tensor = tensor.unsqueeze(0)  # 增加 batch 維度  
        tensor = tensor.to(self.device)  
        return tensor
    
    def inference(self, input):
        results = self._model.predict(source=input, 
                                      imgsz=(self.__model_width, self.__model_height), 
                                      conf=self.__conf_threshold)
        return results

    
    def postprocess(self, output: torch.Tensor) -> dict:  
        return output
    
    def plot(self, output: dict) -> np.ndarray:  
        return output[0].plot()