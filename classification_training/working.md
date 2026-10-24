# YOLO Bug-Crop Classifier Comparison

This experiment compares two pretrained image-classification backbones on the same manually labelled YOLO crops:

- VGG16, using pooled feature maps from five backbone stages instead of its original large fully connected classifier.
- MobileNetV3-Small, using the same five spatial scales and metadata fusion structure.

Both models fuse feature maps from five spatial stages (112, 56, 28, 14, and 7 pixels for a 224-pixel input). Each stage is globally pooled and projected into a compact 64-value vector. The combined visual representation is joined with four metadata features: log bbox width, log bbox height, log aspect ratio, and YOLO confidence. Bounding-box dimensions must be the original YOLO box dimensions, before the crop was expanded by 1.2x. The script does not run YOLO or create crop images.

## CSV manifest

Required columns, using the defaults in the script:

```csv
crop_path,label,source_image_id,bbox_width,bbox_height,yolo_confidence,aspect_ratio
crops/bug_001.jpg,1,frame_001,48,31,0.91,1.5484
crops/bug_002.jpg,other,frame_001,22,25,0.83,0.88
crops/bug_003.jpg,2,frame_002,57,42,0.76,1.3571
```

The rows above only illustrate the schema. Replace paths and values with your actual data. `aspect_ratio` is optional. The label column must contain `1`, `2`, `3`, or `other`. Paths can be absolute or relative to the CSV file's directory. `source_image_id` must identify the original image each detection came from, not the crop filename. If several source images are near-duplicates from the same trap/session, use a stricter session-level group to reduce leakage.

## Run

Install dependencies in an environment with mutually compatible PyTorch and Torchvision builds. The first run downloads the official ImageNet-pretrained model weights if they are not already cached.

```bash
python bug_classifier_compare.py \
  --csv data/bug_crops.csv \
  --output-dir runs/bug_classifier_comparison \
  --group-col source_image_id \
  --warmup-epochs 5 \
  --finetune-epochs 15 \
  --batch-size 16 \
  --num-workers 0
```

For Windows PowerShell, enter the command on one line or replace the backslash line continuations with PowerShell backticks. Keeping `--num-workers 0` is the safer initial choice on Windows.

## Method

- Use a group-aware stratified split so crops from one original image cannot cross train, validation, and test partitions.
- Fit metadata normalization on the training partition only.
- Train the custom head first, then fine-tune only the final backbone block(s) at a lower learning rate.
- Use square padding rather than a random resized crop to avoid cutting off small bug features or distorting the crop's aspect ratio.
- Use square-root inverse-frequency weighted cross-entropy as a conservative class-imbalance baseline.
- Select checkpoints by validation macro-F1, then evaluate the chosen checkpoint on the test partition.

## Outputs

The output directory contains the saved best checkpoint for each backbone, training histories, test predictions and class probabilities, per-class reports, confusion matrices, the fixed split assignment, run configuration, and a summary comparison CSV.

Treat one split as an initial comparison, not a definitive conclusion. With only 500-600 crops, macro-F1 and minority-class recall can vary noticeably with the source-image split. If rare classes appear in only a few original images, collect more independently captured examples or use repeated group-aware cross-validation before drawing strong conclusions.
