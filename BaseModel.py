from abc import ABC, abstractmethod  
from typing import Optional, Any  
from pathlib import Path  
from torch import nn
import os
import torch  
import numpy as np  

class BaseModel(ABC):  
    def __init__(self):  
        self._model: Optional[nn.Module] = None  
        self._is_ready: bool = False  
        self._normalizer = None
        
    @property  
    def is_ready(self) -> bool:  
        return self._is_ready and self._model is not None  
    
    @property  
    def device(self) -> str:  
        return 'cuda' if torch.cuda.is_available() else 'cpu'  
    
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
                if os.path.exists(model_path): 
                    self._model = torch.load(model_path)
                else:
                    raise FileNotFoundError(f"模型檔案不存在：{str(model_path)}")

            self._model.eval()  
            self._model.to(self.device)  
            self._is_ready = True  
            
        except Exception as e:  
            self._is_ready = False  
            raise RuntimeError(f"模型載入失敗：{str(e)}")  
    
    @abstractmethod  
    def _load_model_online(self) -> None:  
        """從網路載入預訓練模型"""  
        pass  
    
    def _validate_image(self, image: np.ndarray) -> None:  
        if not isinstance(image, np.ndarray):  
            raise ValueError("輸入必須是 numpy 陣列")  
        
        if len(image.shape) != 3:  
            raise ValueError(f"影像維度必須為 3，目前為 {len(image.shape)}")  
        
        if image.shape[2] != 3:  
            raise ValueError(f"影像通道數必須為 3，目前為 {image.shape[2]}")  
    
    @abstractmethod  
    def preprocess(self, image: np.ndarray) -> torch.Tensor:  
        """影像預處理"""  
        pass  
    
    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:  
        return self._normalizer(tensor)
    
    @abstractmethod  
    def postprocess(self, output: torch.Tensor) -> Any:  
        """後處理模型輸出"""  
        pass  
    
    @abstractmethod
    def inference(self, input: np.ndarray) -> Any:  
        pass
    
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