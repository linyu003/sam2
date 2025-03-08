from models.pipeline_models import MODEL_SAM2, call_depth_anything,MODEL_MICROSOFT_RESNET_50,MODEL_DEPTH_ANYTHING,call_classification_model,call_blip_captioning,MODEL_BLIP_IMAGE_CAPTIONING
import os

# if using Apple MPS, fall back to CPU for unsupported ops
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cgi
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
from PIL import Image
import io
import json
import notebooks.models.sam2 as sam2


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")



np.random.seed(3)

# 保持 TF32 设置不变
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from models import qwen2_5_vl_awq


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print(f"received get request path: {self.path}, do nothing")
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"msg": "Hello, World!"}).encode("utf-8"))

    def do_POST(self):
        print(f"received post request path: {self.path}")
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
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
            model_name = self.path[len("/call/"):]
            if model_name == MODEL_DEPTH_ANYTHING:
                result = call_depth_anything(image)
            elif model_name == MODEL_MICROSOFT_RESNET_50:
                result = call_classification_model(image)
            elif model_name == MODEL_BLIP_IMAGE_CAPTIONING:
                result = call_blip_captioning(image)
            elif model_name == MODEL_SAM2:
                result = sam2.call_sam2(image)
            elif qwen2_5_vl_awq.ModelName.get(model_name) is not None:
                result = qwen2_5_vl_awq.call_model(model_name, image)
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



def run(
    server_class=HTTPServer, handler_class=RequestHandler, host="0.0.0.0", port=10001
):
    server_address = (host, port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting httpd on port {host}:{port}...")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
