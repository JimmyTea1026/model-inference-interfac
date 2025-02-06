from abc import ABC, abstractmethod  
from typing import Optional, Any  
from pathlib import Path  
from torch import nn
import os
import torch  
import numpy as np  
import cv2

class BaseModel(ABC):  
    def __init__(self):  
        self._model: Optional[nn.Module] = None  
        self._is_ready: bool = False  
        self._normalizer = None
        self._model_width = 0
        self._model_height = 0
        
    @property  
    def is_ready(self) -> bool:  
        return self._is_ready and self._model is not None  
    
    @property  
    def device(self) -> str:  
        return 'cuda' if torch.cuda.is_available() else 'cpu'  
    
    @abstractmethod  
    def _load_model_online(self) -> None:  
        """從網路載入預訓練模型"""  
        pass  
    
    @abstractmethod  
    def _load_model_local(self, model_path) -> None:  
        """從網路載入預訓練模型"""  
        pass  
    
    @abstractmethod  
    def postprocess(self, output: torch.Tensor) -> Any:  
        """後處理模型輸出"""  
        pass  
    
    @abstractmethod
    def inference(self, input: np.ndarray) -> Any:  
        pass
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        '''
        改為等比例縮放影像，剩餘部分補0
        '''  
        resize_image = self._resize_padding(image)
        tensor = torch.from_numpy(resize_image).float() 
        tensor = tensor.permute(2, 0, 1)  # HWC to CHW  
        tensor = tensor.unsqueeze(0)  # 增加 batch 維度  
        tensor = tensor.to(self.device)  
        return tensor

    def _resize_padding(self, image: np.ndarray) -> np.ndarray:
        width_ratio = self._model_width / image.shape[1]
        height_ratio = self._model_height / image.shape[0]
        ratio = min(width_ratio, height_ratio)
        new_width = int(image.shape[1] * ratio)
        new_height = int(image.shape[0] * ratio)
        image = cv2.resize(image, (new_width, new_height))
        top = (self._model_height - new_height) // 2
        bottom = self._model_height - new_height - top
        left = (self._model_width - new_width) // 2
        right = self._model_width - new_width - left
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return image
    
    def load_model(self, model_path: Optional[Path] = None) -> None:  
        """  
        載入模型  

        Args:  
            model_path: 模型權重檔案路徑  

        Raises:  
            FileNotFoundError: 模型檔案不存在  
            RuntimeError: 載入失敗  
        """  
        try:  
            if model_path is None:  
                self._load_model_online()  
            else:  
                self._load_model_local(model_path)

            self._model.eval()  
            self._model.to(self.device)  
            self._is_ready = True  
            
        except Exception as e:  
            self._is_ready = False  
            raise RuntimeError(f"模型載入失敗：{str(e)}")  
      
    def _validate_image(self, image: np.ndarray) -> None:  
        if not isinstance(image, np.ndarray):  
            raise ValueError("輸入必須是 numpy 陣列")  
        
        if len(image.shape) != 3:  
            raise ValueError(f"影像維度必須為 3，目前為 {len(image.shape)}")  
        
        if image.shape[2] != 3:  
            raise ValueError(f"影像通道數必須為 3，目前為 {image.shape[2]}")  
    
    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:  
        return self._normalizer(tensor)
    
    @torch.no_grad()  
    def __call__(self, image: np.ndarray) -> Any:  
        if not self.is_ready:  
            raise RuntimeError("模型尚未準備就緒，請先呼叫 load_model()")  
        
        try:  
            self._validate_image(image)  
            tensor = self.preprocess(image)  
            normalized = self.normalize(tensor)
            output = self.inference(normalized)  
            postprocessed = self.postprocess(output)
            return postprocessed
            
        except Exception as e:  
            raise RuntimeError(f"推論過程發生錯誤：{str(e)}")  