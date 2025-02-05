from transformers import pipeline
from PIL import Image
import requests
from typing import Any, Dict, List
from abc import ABC, abstractmethod
import numpy as np
import io
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch

MODEL_MICROSOFT_RESNET_50 = "microsoft/resnet-50"
MODEL_DEPTH_ANYTHING = "depth-anything/Depth-Anything-V2-Large-hf"

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
            model=MODEL_DEPTH_ANYTHING
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
        包含深度信息的字典，其中depth为PIL Image对象
    """
    global _depth_model
    if _depth_model is None:
        _depth_model = DepthEstimationModel()
    depth_result = _depth_model(image)
    depth_image: Image = depth_result["depth"]

    return {
        "depth": depth_image
    }

def test_depth_anything():
    image = Image.open("notebooks/images/cars.jpg")
    result = call_depth_anything(image)
    
    # 从bytes重新创建图像进行测试
    depth_bytes = result["depth"]
    depth_image = Image.open(io.BytesIO(depth_bytes))
    depth_image.show()
    
    # 转换为numpy数组查看数据
    depth_array = np.asarray(depth_image)
    print(depth_array.shape)
    print(depth_array.dtype)
    print(depth_array)

def test_classification_model():
    image = Image.open("images/cars.jpg")
    results = call_classification_model(image)
    print(results)

class ClassificationModel(BasePipelineModel):
    def _initialize_pipeline(self) -> None:
        self.processor = AutoImageProcessor.from_pretrained(MODEL_MICROSOFT_RESNET_50)
        self.model = ResNetForImageClassification.from_pretrained(MODEL_MICROSOFT_RESNET_50)
    

    def __call__(self, image: Image) -> List[Dict[str, Any]]:
        inputs = self.processor(image, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits

        # 计算所有类别的概率
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        # 获取概率大于等于10%的索引
        threshold = 0.10  # 10%
        mask = probabilities >= threshold
        filtered_probs = probabilities[mask]
        filtered_indices = torch.nonzero(mask).squeeze()
        
        # 按概率降序排序
        sorted_indices = torch.argsort(filtered_probs, descending=True)
        
        # 构建返回结果
        results = []
        for i, idx in enumerate(sorted_indices):
            prob = filtered_probs[idx]
            class_idx = filtered_indices[idx]
            label = self.model.config.id2label[class_idx.item()]
            results.append({
                "label": label,
                "score": prob.item()
            })
        
        return results

# 全局单例实例
_classification_model = None

def call_classification_model(image: Image) -> List[Dict[str, Any]]:
    """
    调用分类模型的全局函数
    
    Args:
        image: PIL Image对象
    Returns:
        [
            {
                "label": str,
                "score": float
            },
            ...
        ]
    """
    global _classification_model
    if _classification_model is None:
        _classification_model = ClassificationModel()
    return _classification_model(image)

if __name__ == "__main__":
    test_classification_model()

