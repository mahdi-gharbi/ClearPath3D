"""Pretrained 2D instance segmentation (torchvision Mask R-CNN, COCO) as an optional ``Detector``.

Optional dependency: ``torch`` + ``torchvision`` (CPU wheels). Weights are downloaded once by torchvision
from download.pytorch.org into the torch hub cache. COCO classes only; a label is the detector's class
name, never inferred from geometry. Import is lazy so the core never requires torch.
"""

from __future__ import annotations

import numpy as np

from clearpath3d.contracts import RGBDFrame
from clearpath3d.perception.semantics import Detection2D


class TorchvisionMaskRCNN:
    name = "torchvision_maskrcnn_resnet50_fpn_coco"

    def __init__(
        self, score_threshold: float = 0.6, mask_threshold: float = 0.5, num_threads: int = 4
    ):
        import torch
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_Weights,
            maskrcnn_resnet50_fpn,
        )

        torch.set_num_threads(num_threads)
        self.weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn(weights=self.weights).eval()
        self.categories = self.weights.meta["categories"]
        self.transform = self.weights.transforms()
        self.score_threshold, self.mask_threshold = score_threshold, mask_threshold
        self._torch = torch

    def detect(self, frame: RGBDFrame) -> list[Detection2D]:
        torch = self._torch
        img = torch.from_numpy(np.ascontiguousarray(frame.color_rgb)).permute(2, 0, 1)
        with torch.inference_mode():
            out = self.model([self.transform(img)])[0]
        dets = []
        for box, label, score, mask in zip(
            out["boxes"], out["labels"], out["scores"], out["masks"], strict=True
        ):
            s = float(score)
            if s < self.score_threshold:
                continue
            u0, v0, u1, v1 = (int(round(float(x))) for x in box)
            m = (mask[0] >= self.mask_threshold).numpy()
            dets.append(
                Detection2D(self.categories[int(label)], s, (u0, v0, u1, v1), m, source=self.name)
            )
        return dets
