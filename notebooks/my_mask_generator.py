# In[0]
import os
import threading

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cgi
import base64

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

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.utils.common_util import serialize_ndarray
import time



class Sam2_Model:
    def __init__(self,ckpt_path:str,model_cfg_path:str,device:torch.device):
        self.sam2 = build_sam2(ckpt_path, model_cfg_path, device=device, apply_postprocessing=False)
        self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2)

    def gen_masks(self, image: Image):
        image = np.array(image.convert("RGB"))
        start_time = time.time()
        masks = self.mask_generator.generate(image)
        print(f"Time taken: {time.time() - start_time} seconds")
        return masks



_sam2_model = None
lock_for_init = threading.Lock()

def init_sam2_model():
    if _sam2_model is None:
        with lock_for_init:
            if _sam2_model is None:
                sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
                device = torch.device("cuda")
                global _sam2_model
                _sam2_model = Sam2_Model(sam2_checkpoint, model_cfg, device)

def call_sam2(image: Image):
    init_sam2_model()
    result =  _sam2_model.gen_masks(image)
    for r in result:
        seg: np.ndarray = r["segmentation"]
        r["segmentation"] = serialize_ndarray(seg)
    return result






