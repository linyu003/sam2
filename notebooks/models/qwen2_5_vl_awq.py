from datetime import datetime
from pathlib import Path
import time
import uuid
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import Any, Dict, List
from PIL import Image

model_name = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained(model_name)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct-AWQ", min_pixels=min_pixels, max_pixels=max_pixels)

def call_qwen2_5_vl_awq(image: Image.Image) -> Dict[str, Any]:
    global processor
    # datetime + randatetimeng
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
            {"type": "text", "text": "Describe this image with no more than 20 words."},
        ],
    }
]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )   
    return {
        "description_sentence": output_text
    }

if __name__ == "__main__":
    for i in range(10):
        start_time = time.time()
        image = Image.open("qwen_images/groceries.jpg")
        print(call_qwen2_5_vl_awq(image))
        time_cost = time.time() - start_time
        print(f"time cost: {time_cost} seconds for {i+1} times")
