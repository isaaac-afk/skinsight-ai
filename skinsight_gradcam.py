"""
Grad-CAM explainability layer for SkinSight AI.

Sits on top of the trained EfficientNet-B0 classifier (see skinsight_model.py).
For a given lesion image it produces:
  - the model's ranked predictions (via the existing classifier)
  - a heatmap over the image showing which regions drove the top prediction

This is the "honesty about limits" feature: instead of an opaque label,
the user sees *what the model looked at*. If the heatmap lights up on a
ruler marking, hair, or skin background instead of the lesion, that's a
visible signal the prediction is untrustworthy.

NOT a diagnostic tool. Educational/informational use only.

Deps: torch, torchvision, timm, pillow, numpy, matplotlib (for the overlay).
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm

from skinsight_model import build_transforms, MEAN, STD, DEVICE, CKPT_PATH
import timm


class GradCAM:
    """
    Grad-CAM for a CNN classifier.

    Registers hooks on a target convolutional layer to capture its
    activations (forward) and gradients (backward), then weights the
    activation maps by the mean gradient to produce a class-discriminative
    heatmap.
    """

    def __init__(self, ckpt_path: str = CKPT_PATH, target_layer: str | None = None):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        self.classes = ckpt["classes"]
        self.model = timm.create_model(
            ckpt["model_name"], pretrained=False, num_classes=len(self.classes)
        ).to(DEVICE)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        _, self.tf = build_transforms()

        # For EfficientNet in timm, the last conv stage is `conv_head`.
        # Allow override for experimenting with earlier layers.
        self.target_layer = self._resolve_layer(target_layer or "conv_head")

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._register_hooks()

    def _resolve_layer(self, name: str):
        module = dict(self.model.named_modules()).get(name)
        if module is None:
            raise ValueError(
                f"Layer '{name}' not found. Available conv-ish layers include: "
                f"{[n for n, _ in self.model.named_modules() if 'conv' in n][:10]}"
            )
        return module

    def _register_hooks(self):
        def fwd_hook(_module, _inp, output):
            self._activations = output.detach()

        def bwd_hook(_module, _grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        # full_backward_hook is the non-deprecated API
        self.target_layer.register_full_backward_hook(bwd_hook)

    def _preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        return self.tf(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)

    def __call__(self, pil_image: Image.Image, class_idx: int | None = None):
        """
        Returns:
            cam:        (H, W) float array in [0, 1], same size as model input
            ranked:     list of (label, prob) ranked descending
            target_idx: the class index the CAM explains
        """
        x = self._preprocess(pil_image)
        x.requires_grad_(True)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]

        if class_idx is None:
            class_idx = int(probs.argmax().item())

        # Backprop the chosen class score to populate gradients.
        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=False)

        acts = self._activations[0]        # (C, h, w)
        grads = self._gradients[0]          # (C, h, w)
        weights = grads.mean(dim=(1, 2))    # (C,) — global-avg-pooled gradients

        cam = torch.relu((weights[:, None, None] * acts).sum(0))  # (h, w)

        # Upsample to input resolution and normalize to [0, 1].
        cam = cam[None, None]
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        ranked = sorted(
            zip(self.classes, probs.detach().cpu().tolist()),
            key=lambda p: p[1], reverse=True,
        )
        return cam.cpu().numpy(), ranked, class_idx

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------
    def _denormalize(self, x: torch.Tensor) -> np.ndarray:
        mean = torch.tensor(MEAN).view(3, 1, 1)
        std = torch.tensor(STD).view(3, 1, 1)
        img = (x.cpu() * std + mean).clamp(0, 1)
        return img.permute(1, 2, 0).numpy()

    def overlay(self, pil_image: Image.Image, cam: np.ndarray,
                alpha: float = 0.45) -> Image.Image:
        """Blend the heatmap over the (resized) input image for display."""
        x = self._preprocess(pil_image)[0]
        base = self._denormalize(x)                       # (H, W, 3) in [0,1]
        heat = cm.jet(cam)[..., :3]                        # (H, W, 3) RGB
        blended = (1 - alpha) * base + alpha * heat
        blended = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(blended)

    def explain(self, pil_image: Image.Image, class_idx: int | None = None):
        """
        One-call convenience method for the Flask backend.
        Returns predictions, the overlay image, and a trust signal.
        """
        cam, ranked, idx = self(pil_image, class_idx)
        overlay_img = self.overlay(pil_image, cam)

        # Crude out-of-distribution / trust heuristic: how concentrated is the
        # attention? A CAM that's diffuse across the whole frame (high entropy)
        # often means the model isn't focusing on a lesion at all.
        p = cam / (cam.sum() + 1e-8)
        entropy = float(-(p * np.log(p + 1e-12)).sum())
        max_entropy = float(np.log(cam.size))
        focus_score = round(1.0 - entropy / max_entropy, 3)  # 1 = tightly focused

        return {
            "predictions": [{"label": c, "confidence": round(pr, 4)} for c, pr in ranked],
            "top_label": ranked[0][0],
            "explained_class": self.classes[idx],
            "focus_score": focus_score,
            "overlay": overlay_img,  # PIL Image; encode to base64/PNG in the API
            "disclaimer": (
                "Informational only — NOT a medical diagnosis. The heatmap shows "
                "where the model focused; if it is not on the lesion, treat the "
                "result as unreliable. Consult a dermatologist for any skin concern."
            ),
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python skinsight_gradcam.py <image_path> [out_path]")
        raise SystemExit(1)

    img = Image.open(sys.argv[1])
    cam_engine = GradCAM()
    result = cam_engine.explain(img)

    out_path = sys.argv[2] if len(sys.argv) > 2 else "gradcam_overlay.png"
    result["overlay"].save(out_path)

    print(f"Explained class: {result['explained_class']}")
    print(f"Focus score:     {result['focus_score']}  (1.0 = tightly focused on a region)")
    print("Top predictions:")
    for p in result["predictions"][:3]:
        print(f"  {p['label']:<24} {p['confidence']:.3f}")
    print(f"\nOverlay saved to {out_path}")
    print(result["disclaimer"])
