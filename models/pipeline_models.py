from transformers import pipeline
from PIL import Image
import requests
from typing import Any, Dict
from abc import ABC, abstractmethod
import numpy as np

class BasePipelineModel(ABC):
    def __init__(self):
        self.pipe = None
        self._initialize_pipeline()
    
    @abstractmethod
    def _initialize_pipeline(self) -> None:
        """初始化特定的pipeline"""
        pass
    
    @abstractmethod
    def __call__(self, image: Image) -> Dict[str, Any]:
        """处理输入图像并返回结果"""
        pass

class DepthEstimationModel(BasePipelineModel):
    def _initialize_pipeline(self) -> None:
        self.pipe = pipeline(
            task="depth-estimation", 
            model="depth-anything/Depth-Anything-V2-Large-hf"
        )
    
    def __call__(self, image: Image) -> Dict[str, Any]:
        return self.pipe(image)

# 全局单例实例
_depth_model = None

def call_depth_anything(image: Image) -> Dict[str, Any]:
    """
    调用深度估计模型的全局函数
    
    Args:
        image: PIL Image对象
    Returns:
        包含深度信息的字典
    """
    global _depth_model
    if _depth_model is None:
        _depth_model = DepthEstimationModel()
    return {
        "depth": np.asarray(_depth_model(image)["depth"]),
    }


if __name__ == "__main__":
    image = Image.open("notebooks/images/cars.jpg")
    result = call_depth_anything(image)
    i:Image = result["depth"]
    i.show()
    depth_array = np.asarray(result["depth"])
    print(depth_array.shape)
    print(depth_array.dtype)
    print(depth_array)
