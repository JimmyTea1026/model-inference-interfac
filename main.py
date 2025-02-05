from YoloModel import YOLOModel
import numpy as np
import cv2

def main():  
    try:  
        # 初始化模型  
        config = {"height": 640, "width": 640, "conf_threshold": 0.1}
        model = YOLOModel(**config)  
        
        # 載入模型  
        model.load_model()  
        
        # 準備測試影像  
        image = cv2.imread("yolo_sample.jpg")
        
        # 執行推論  
        results = model(image)  
        
        annotated_image = model.plot(results)
        
        cv2.imshow("Annotated Image", annotated_image)
        cv2.waitKey(0)
        
    except Exception as e:  
        print(f"錯誤：{str(e)}")  

if __name__ == "__main__":  
    main()