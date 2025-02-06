import torch  
import numpy as np
import torch
import cv2
import os
from ultralytics import YOLO
from BaseModel import BaseModel
from Normalizer import Normalizer

class YOLOModel(BaseModel):      
    def __init__(self, height: int = 640, width: int = 640, conf_threshold: float = 0.2, normalizer=None):  
        super().__init__()
        self._model_height = height
        self._model_width = width
        self.__conf_threshold = conf_threshold
        self._normalizer = normalizer
    
    def _load_model_local(self, model_path):  
        """  
        從本地載入 YOLO 模型  
        
        Args:  
            model_path (str): 模型文件路徑  
        """  
        if not os.path.exists(model_path):  
            raise FileNotFoundError(f"模型文件不存在：{model_path}")  
        self._model = YOLO(model_path)  
    
    def _load_model_online(self):
        # 從網路載入預訓練模型，載入yolov10模型
        self._model = YOLO("yolov10n.pt")
        return 
    
    @torch.no_grad() 
    def inference(self, input_data):
        results = self._model.predict(source=input_data, 
                                      imgsz=(self._model_width, self._model_height), 
                                      conf=self.__conf_threshold)
        return results

    
    def postprocess(self, output: torch.Tensor) -> dict:  
        return output
    
    def plot(self, output: dict) -> np.ndarray:  
        return output[0].plot()
    
    
def main():  
    try:  
        # 初始化模型  
        config = {"height": 640, "width": 640, "conf_threshold": 0.25, "normalizer": Normalizer("yolo")}
        model = YOLOModel(**config)  
        
        # 載入模型  
        model.load_model("./assets/yolov10n.pt")  
        
        # 讀取影片
        cap = cv2.VideoCapture("./assets/demo.mp4")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(image)
            annotated_image = model.plot(results)
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imshow("Annotated Image", annotated_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:  
        print(f"錯誤：{str(e)}")  

if __name__ == "__main__":  
    main()