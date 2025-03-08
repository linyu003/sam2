# In[0]
import os

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.pipeline_models import call_depth_anything,MODEL_MICROSOFT_RESNET_50,MODEL_DEPTH_ANYTHING,call_classification_model,call_blip_captioning,MODEL_BLIP_IMAGE_CAPTIONING
from PIL import Image
import cgi
import base64
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # 创建两个上下文管理器，一个用于 SAM2，一个用于 depth-anything
    sam2_autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    depth_autocast = torch.autocast("cuda", dtype=torch.float32)
    
    # 只为 SAM2 启用 bfloat16
    sam2_autocast.__enter__()
    
    # 保持 TF32 设置不变
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

np.random.seed(3)

from models import qwen2_5_vl_awq


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2

            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Try to smooth contours
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


# plt.figure(figsize=(20, 20))
# plt.imshow(image)
# plt.axis('off')
# plt.show()


from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.utils.common_util import serialize_ndarray
import time

sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2)


def gen_masks(image: Image):
    # image = Image.open('images/cars.jpg')
    image = np.array(image.convert("RGB"))
    start_time = time.time()
    masks = mask_generator.generate(image)
    print(f"Time taken: {time.time() - start_time} seconds")
    return masks


from http.server import BaseHTTPRequestHandler, HTTPServer
from PIL import Image
import io
import json


class RequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        print(f"received get request path: {self.path}, do nothing")

    def do_POST(self):
        print(f"received post request path: {self.path}")
        if self.path in ["/gen_mask"] or self.path.startswith("/call/"):
            print(f"/gen_mask processing")
            # 解析Content-Length头
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            # 解析multipart/form-data格式的数据
            form = cgi.FieldStorage(
                fp=io.BytesIO(post_data),
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers["Content-Type"],
                },
            )

            # 获取文件内容
            if "image" in form:
                image_file = form["image"].file
                image = Image.open(image_file)

                # 根据不同的路径使用不同的上下文管理器
                if self.path == "/gen_mask":
                    result = gen_masks(image)
                    for r in result:
                        seg: np.ndarray = r["segmentation"]
                        r["segmentation"] = serialize_ndarray(seg)
                else:
                    model_name = self.path[len("/call/"):]
                    if model_name == MODEL_DEPTH_ANYTHING:
                        # 对 depth-anything 使用 float32
                        with depth_autocast:
                            result = call_depth_anything(image)


                        img_byte_arr = io.BytesIO()
                        result["depth"].save(img_byte_arr, format='PNG')
                        
                        result["depth"] = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                    elif model_name == MODEL_MICROSOFT_RESNET_50:
                        result = call_classification_model(image)
                    elif model_name == MODEL_BLIP_IMAGE_CAPTIONING:
                        result ={
                            "generated_text": call_blip_captioning(image)
                        }
                    elif model_name == qwen2_5_vl_awq.model_name:
                        result = qwen2_5_vl_awq.call_qwen2_5_vl_awq(image)
                    else:
                        raise ValueError(f"Unknown model name: {model_name}")




                # 将结果转换为JSON格式
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            else:
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"error": "No image provided"}).encode("utf-8")
                )
        else:
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")


def run(
    server_class=HTTPServer, handler_class=RequestHandler, host="0.0.0.0", port=10001
):
    server_address = (host, port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting httpd on port {host}:{port}...")
    httpd.serve_forever()


if __name__ == "__main__":
    run()

