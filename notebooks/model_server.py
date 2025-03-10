from ast import Dict
import sys
from typing import Any, List
from models.pipeline_models import MODEL_SAM2, call_depth_anything,MODEL_MICROSOFT_RESNET_50,MODEL_DEPTH_ANYTHING,call_classification_model,call_blip_captioning,MODEL_BLIP_IMAGE_CAPTIONING
from models import pipeline_models
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
        form:cgi.FieldStorage = cgi.FieldStorage(
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
            params = json.loads(form.getvalue("params", "{}"))
            model_name = self.path[len("/call/"):]
            if model_name == MODEL_DEPTH_ANYTHING:
                result = call_depth_anything(image)
            elif model_name == MODEL_MICROSOFT_RESNET_50:
                result = call_classification_model(image)
            elif model_name == MODEL_BLIP_IMAGE_CAPTIONING:
                result = call_blip_captioning(image)
            elif model_name == MODEL_SAM2:
                import notebooks.models.sam2 as sam2
                result = sam2.call_sam2(image)
            elif qwen2_5_vl_awq.ModelName.get(model_name) is not None:
                result = qwen2_5_vl_awq.call_model(model_name, image, **params)
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
def parse_params(argv:list[str]) -> dict[str, Any]:
    import argparse
    parser = argparse.ArgumentParser(description='启动模型服务器')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器主机名')
    parser.add_argument('--port', type=int, default=10001, help='服务器端口号')
    parser.add_argument('--init_models', nargs='+', default=[], help='初始化模型')
    args = parser.parse_args(argv[1:])
    return args

def run(
    server_class=HTTPServer, 
    handler_class=RequestHandler, 
):
    params = parse_params(sys.argv)
    print(f"params: {params}")
    for model in params.init_models:
        if qwen2_5_vl_awq.ModelName.get(model) is not None:
            qwen2_5_vl_awq.get_model(qwen2_5_vl_awq.ModelName.get(model))
        elif model == MODEL_SAM2:
            import notebooks.models.sam2 as sam2
            sam2.init_sam2_model()
        elif model == MODEL_DEPTH_ANYTHING:
            pipeline_models.init_depth_model()
        elif model == MODEL_MICROSOFT_RESNET_50:
            pipeline_models.init_classification_model()
        elif model == MODEL_BLIP_IMAGE_CAPTIONING:
            pipeline_models.init_blip_model()
        else:
            raise ValueError(f"Unknown model name: {model}")
    server_address = (params.host, params.port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting httpd on port {params.host}:{params.port}...")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
