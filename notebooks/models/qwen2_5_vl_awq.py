from datetime import datetime
from enum import Enum
from pathlib import Path
import time
import uuid
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import Any, Dict, List
from PIL import Image
import logging
from threading import Lock
log = logging.getLogger(__name__)

# default: Load the model on the available device(s)
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = A

# 定义一个模型名的枚举
class ModelName(Enum):
    QWEN2_5_VL_3B_INSTRUCT = "Qwen/Qwen2.5-VL-3B-Instruct"
    QWEN2_5_VL_7B_INSTRUCT = "Qwen/Qwen2.5-VL-7B-Instruct"
    QWEN2_5_VL_14B_INSTRUCT = "Qwen/Qwen2.5-VL-14B-Instruct"
    QWEN2_5_VL_32B_INSTRUCT = "Qwen/Qwen2.5-VL-32B-Instruct"
    QWEN2_5_VL_3B_INSTRUCT_AWQ = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
    QWEN2_5_VL_7B_INSTRUCT_AWQ = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    QWEN2_5_VL_14B_INSTRUCT_AWQ = "Qwen/Qwen2.5-VL-14B-Instruct-AWQ"
    QWEN2_5_VL_32B_INSTRUCT_AWQ = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"

    @staticmethod
    def get(model_name:str) -> 'ModelName':
        if model_name in ModelName.__members__:
            return ModelName[model_name]
        for model in ModelName:
            if model_name == model.value:
                return model
        return None

class Qwen2_5_VL:
    def __init__(self, model_name: ModelName):
        self.model_name:ModelName = model_name
        self.model:Qwen2_5_VLForConditionalGeneration = None
        self.processor:AutoProcessor = None

    def load_model(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name.value, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name.value)
    
    def call_model(self, image: Image.Image, prompt:str = None) -> Dict[str, Any]:
        if prompt is None or prompt.strip() == "":
            prompt = "Describe this image with no more than 20 words."

        if self.model is None or self.processor is None:
            log.warn("model or processor is not loaded")
            self.load_model()

        image_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4()}.jpg"
        p=Path("qwen_images")
        p.mkdir(parents=True, exist_ok=True)
        image_path = p / image_name
        image.save(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                        "image": f"file://{image_path.as_posix()}"
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )   
        return {
            "description_sentence": output_text
        }

model_map = {}
lock_for_model_map = Lock()

def get_model(model_name:ModelName) -> Qwen2_5_VL:
    if model_name in model_map:
        return model_map[model_name]
    with lock_for_model_map:
        if model_name in model_map:
            return model_map[model_name]
        model = Qwen2_5_VL(model_name)
        model.load_model()
        model_map[model_name] = model
        return model
def call_model(model_name:str|ModelName, image:Image.Image, prompt:str = None) -> Dict[str, Any]:
    if isinstance(model_name, str):
        model_name = ModelName.get(model_name)
    model_name:ModelName
    model = get_model(model_name)
    return model.call_model(image, prompt)

if __name__ == "__main__":
    for i in range(10):
        start_time = time.time()
        image = Image.open("images/groceries.jpg")
        print(call_model(ModelName.QWEN2_5_VL_7B_INSTRUCT, image))
        time_cost = time.time() - start_time
        print(f"time cost: {time_cost} seconds for {i+1} times")
