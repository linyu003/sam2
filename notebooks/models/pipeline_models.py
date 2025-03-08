import base64
from transformers import pipeline
from PIL import Image
import requests
from typing import Any, Dict, List
from abc import ABC, abstractmethod
import numpy as np
import io
from transformers import AutoImageProcessor, ResNetForImageClassification
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import threading

MODEL_MICROSOFT_RESNET_50 = "microsoft/resnet-50"
MODEL_DEPTH_ANYTHING = "depth-anything/Depth-Anything-V2-Large-hf"
MODEL_BLIP_IMAGE_CAPTIONING = "Salesforce/blip-image-captioning-base"
MODEL_SAM2 = "sam2"
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
        self.autocast = torch.autocast("cuda", dtype=torch.float32)
    
    def __call__(self, image: Image) -> Dict[str, Any]:
        with self.autocast:
            return self.pipe(image)

# 全局单例实例
_depth_model = None

lock_for_init = threading.Lock()

def init_depth_model():
    global _depth_model
    if _depth_model is None:
        with lock_for_init:
            if _depth_model is None:
                _depth_model = DepthEstimationModel()

def call_depth_anything(image: Image) -> Dict[str, Any]:
    global _depth_model
    init_depth_model()
    depth_result = _depth_model(image)
    depth_image: Image = depth_result["depth"]
    result =  {
        "depth": depth_image
    }
    img_byte_arr = io.BytesIO()
    result["depth"].save(img_byte_arr, format='PNG')
    result["depth"] = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    return result

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
        
        # 如果没有预测结果超过阈值，返回概率最高的3个结果
        if not mask.any():
            top_k = 3
            top_probs, top_indices = torch.topk(probabilities, k=top_k)
            return [{
                "label": self.model.config.id2label[idx.item()],
                "score": prob.item()
            } for prob, idx in zip(top_probs, top_indices)]
            
        filtered_probs = probabilities[mask]
        filtered_indices = torch.nonzero(mask).squeeze()
        
        # 修改处理单个结果的情况
        if filtered_probs.numel() == 1:  # 使用numel()替代dim()检查
            return [{
                "label": self.model.config.id2label[filtered_indices.item()],
                "score": filtered_probs.item()
            }]
        
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

lock_for_init = threading.Lock()

def init_classification_model():
    global _classification_model
    if _classification_model is None:
        with lock_for_init:
            if _classification_model is None:
                _classification_model = ClassificationModel()

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
    init_classification_model()

    return _classification_model(image)
class BlipCaptioningModel:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

    def __call__(self, image: Image) -> str:
        # unconditional image captioning
        inputs = self.processor(image, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)

# 全局单例实例
_blip_model = None
lock_for_init = threading.Lock()

def init_blip_model():
    global _blip_model
    if _blip_model is None:
        with lock_for_init:
            if _blip_model is None:
                _blip_model = BlipCaptioningModel()


def call_blip_captioning(image: Image) -> str:
    """
    调用BLIP图像描述模型的全局函数
    
    Args:
        image: PIL Image对象
    Returns:
        str: 生成的图像描述文本
    """
    global _blip_model
    init_blip_model()
    return {
        "generated_text": _blip_model(image)
    }

if __name__ == "__main__":
    # test_classification_model()
    call_blip_captioning(image=None)

