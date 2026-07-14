# SkinSight AI

An educational dermatology image classifier that shows its work. It predicts a lesion class from the ISIC 2019 taxonomy and, for every prediction, renders a Grad-CAM heatmap of where the network looked plus a focus score indicating whether that attention is concentrated enough to trust.
> Not a medical device. Outputs are informational classifications for learning and demonstration, not diagnoses. Always consult a dermatologist for any skin concern.

## Why this exists
Most "skin cancer classifier" projects report a single accuracy number and stop there. That number hides the failure mode that matters: a confident prediction made for the wrong reasons - keying off a ruler marking, hair, or background skin instead of the lesion. This project treats **trustworthiness as a first-class output**, not an afterthought. The model is paired with an explainability layer so a human can see *what* drove each prediction and decide whether to believe it.

## What it does
- Classifies a lesion image across the 8 ISIC 2019 diagnostic categories (melanoma, melanocytic nevus, basal cell carcinoma, actinic keratosis, benign keratosis, dermatofibroma, vascular lesion, squamous cell carcinoma).
- Generates a **Grad-CAM** heatmap over the input showing the regions that most influenced the top prediction.
- Computes a focus score a heuristic measuring how concentrated the model''s attention is. Diffuse attention (often a sign the image is off-target or out-of-distribution) triggers a visible low-trust warning.
- Serves all of this through a Flask API and a single-page web demo.

## Approach
- **Transfer learning** on EfficientNet-B0 (pretrained on ImageNet) rather than a from-scratch CNN - far stronger on a dataset this size while staying lightweight.
- **Class imbalance handled twice**: ISIC is dominated by benign nevi, so training uses both a weighted random sampler and a class-weighted loss. Model selection is on **macro-AUC**, not accuracy.
- **Grad-CAM** hooks the final convolutional layer to weight activation maps by their gradients, producing a class-discriminative heatmap.

## Results

On a balanced subset (~1,500 images/class, 20% held out for validation), the model reaches a macro-AUC in the ~0.94 range - comparable to published ISIC baselines for this backbone.

## Honest limitations

- The **focus score is a crude entropy heuristic**. It reliably flags genuinely off-target images, but it under-scores legitimate lesions that fill most of the frame. A size-normalized metric would behave more intuitively - a planned improvement.
- Trained on a subset, not the full archive, for tractable CPU training.
- ISIC''s dermoscopic images differ from consumer phone photos and skew toward lighter skin tones; this model would not generalize reliably to either, which is one reason the project is framed as educational rather than diagnostic.

## Stack

Python, PyTorch, timm (EfficientNet-B0), Flask, vanilla JS/HTML front-end

## Running it

    pip install -r requirements.txt

    # 1. Prepare data (download ISIC 2019 Training Input + GroundTruth CSV into data/):
    python prepare_isic2019.py --csv data/ISIC_2019_Training_GroundTruth.csv --images data/ISIC_2019_Training_Input --out data/isic --per-class 1500

    # 2. Train (saves skinsight_efficientnet_b0.pt on best validation macro-AUC):
    python skinsight_model.py

    # 3. Run the demo, then open http://127.0.0.1:5000
    python app.py

The ISIC 2019 dataset (CC-BY-NC) is not included in this repo and must be downloaded separately from the ISIC Challenge archive (https://challenge.isic-archive.com/).

## License / data note

Code is provided for educational use. The ISIC 2019 data is licensed CC-BY-NC (non-commercial); it is not redistributed here.
