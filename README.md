# Space Debris Detection Using Deep Learning and Exploratory Data Analysis

An EDA-driven computer vision project for detecting space debris in
annotated space imagery using **YOLOv9**.

## Overview

Space debris detection is challenging because debris objects can be
tiny, low-contrast, and distributed unpredictably across dark space
backgrounds. This project combines exploratory data analysis (EDA) with
a YOLO-based object detection pipeline to identify debris and evaluate
detection performance.

The project analyzes object sizes, bounding-box geometry, object counts,
and spatial distribution, then uses those observations to inform
preprocessing, augmentation, and model design.

## Objectives

-   Analyze the space-debris dataset and its annotation characteristics.
-   Detect debris objects and localize them with bounding boxes.
-   Investigate challenges involving small objects, low contrast, and
    dense scenes.
-   Evaluate detection using precision, recall, F1-score, and mean
    Average Precision (mAP).
-   Compare the YOLOv9 approach with a YOLOv5 baseline.

## Dataset

The report describes a Kaggle dataset originating from the AIcrowd
space-debris detection challenge.

  -------------------------------------------------------------------------------------------------------------------
  Property                            Description
  ----------------------------------- -------------------------------------------------------------------------------
  Source                              [Kaggle --- Space Debris Detection
                                      Dataset](https://www.kaggle.com/datasets/sadianawar/debris-detection-dataset)

  Approximate corpus size             27,000 images

  Reported split                      20,000 training, 2,000 validation, 5,000 test images

  Image format                        JPEG

  Image resolution                    256 × 256 pixels in the source dataset

  Annotation type                     Bounding boxes

  Annotation format in source CSV     `[xmin, xmax, ymin, ymax]`
  -------------------------------------------------------------------------------------------------------------------

The report also describes **754 images manually annotated using
Roboflow**. These annotations were converted into a YOLO-compatible
format and used in the model-development pipeline. The report mentions
integrating these annotations with the larger image corpus.

> **Dataset note:** Check the actual dataset files and split
> configuration before reproducing the experiment. The report describes
> both a 27,000-image corpus and a 754-image manually annotated subset;
> these figures refer to different stages of the workflow.

## Exploratory Data Analysis (EDA)

The EDA examines the following properties:

-   **Bounding-box width and height:** The dataset contains a range of
    object sizes, with many small objects.
-   **Objects per image:** Most analyzed images contain approximately
    3--5 debris objects.
-   **Aspect ratios:** The report observes that many boxes have aspect
    ratios around 0.8--0.95, indicating roughly compact objects.
-   **Width--height relationship:** Bounding-box width and height show a
    strong positive relationship.
-   **Object density versus size:** Images with more objects tend to
    contain smaller debris.
-   **Spatial distribution:** Objects appear across the image rather
    than in a consistent fixed location.

### Modeling implications

These observations motivate high-resolution feature extraction,
multi-scale detection, careful localization, image augmentation, and
evaluation that pays particular attention to small or low-contrast
objects.

## Methodology

The workflow described in the report is:

1.  **Dataset collection:** Obtain space imagery from Kaggle.
2.  **Annotation:** Manually annotate a subset of 754 images using
    Roboflow.
3.  **Preprocessing:** Apply auto-orientation, grayscale conversion,
    auto-contrast, and filtering of unannotated images.
4.  **Resizing and normalization:** Resize model inputs to 640 × 640
    pixels and normalize pixel values.
5.  **Region enhancement:** Use static cropping, dynamic cropping, and 2
    × 2 tiling to help expose small objects.
6.  **Data augmentation:** Apply geometric and photometric
    transformations, along with noise, blur, and cutout.
7.  **Label conversion:** Export annotations in YOLO format with
    normalized bounding boxes.
8.  **Training:** Train YOLOv9 using GPU acceleration in Google Colab.
9.  **Inference:** Predict bounding boxes and confidence scores in a
    forward pass.
10. **Post-processing:** Apply Non-Maximum Suppression (NMS) to remove
    redundant overlapping detections.
11. **Evaluation:** Analyze precision, recall, F1-score, and mAP.

## Model

The proposed detector is **YOLOv9**. The report highlights two
architectural ideas:

-   **GELAN (Generalized Efficient Layer Aggregation Network):**
    Supports feature reuse and aims to preserve useful spatial
    information.
-   **PGI (Programmable Gradient Information):** Helps retain richer
    gradient information during training.

The report describes training for **50 epochs** using bounding-box
regression loss, classification loss, and Distribution Focal Loss (DFL).
Google Colab GPU acceleration is used for training.

## Technologies and Tools

-   **Python** --- data analysis and model workflow
-   **YOLOv9** --- object detection
-   **YOLOv5** --- baseline comparison
-   **Roboflow** --- image annotation and preprocessing
-   **Google Colab** --- GPU-assisted training
-   **Pandas / NumPy** --- data and annotation processing
-   **Matplotlib** --- EDA and training-curve visualization
-   **Computer vision and deep learning** --- detection, augmentation,
    and evaluation

The exact package versions and complete dependency list are not
specified in the report.

## Reported Results

The following values are reported in the project paper. They should be
treated as **reported experimental results**, not independently
reproduced benchmarks.

  Metric                     YOLOv9 (Version 2)   YOLOv5 (Version 1)
  ------------------------ -------------------- --------------------
  Accuracy (as reported)                 \~0.95               \~0.90
  Precision                                0.95                0.895
  Recall                                   0.94                 0.87
  F1-score                                 0.95                 0.91
  mAP@0.5                                  0.97                 0.95
  mAP@0.5:0.95                             0.43         Not reported

The report describes decreasing training loss and stable performance
across epochs. It also notes that performance is weaker for extremely
small objects, especially when localization is difficult.

### Interpretation

-   **Precision (\~0.95):** Most predicted detections are reported as
    correct.
-   **Recall (\~0.94):** The model detects a large proportion of labeled
    debris objects, although some may be missed.
-   **F1-score (\~0.95):** The reported precision and recall are well
    balanced.
-   **mAP@0.5 (\~0.97):** The reported detection performance is strong
    at an IoU threshold of 0.5.
-   **mAP@0.5:0.95 (\~0.43):** Performance is lower under stricter IoU
    thresholds, suggesting that precise box localization remains
    challenging.

## Limitations

The report identifies several limitations:

-   Extremely small or sub-pixel debris can be difficult to detect.
-   Low signal-to-noise ratios and weak object contrast make visual
    features harder to distinguish.
-   Dense scenes can make localization ambiguous.
-   A single dataset may not represent every imaging condition.
-   Real-time onboard deployment would require further latency and
    hardware evaluation.
-   The report does not provide a complete reproducibility package with
    pinned dependencies, exact training configuration, and model
    weights.

## Future Work

Potential next steps identified in the report include:

-   Improve detection of small debris objects.
-   Investigate transformer-based detection architectures.
-   Integrate real-time detection systems.
-   Evaluate on more diverse datasets and imaging conditions.

## Reproducibility and Usage

This README documents the workflow and results described in the project
report. It does **not** claim that the model can be trained or run
directly from this repository: the report alone does not include a
verified repository structure, complete installation instructions,
trained weights, or an inference script.

To make the project reproducible, add the following artifacts to the
repository:

-   Dataset access instructions and license information
-   Annotation conversion and preprocessing scripts
-   A pinned `requirements.txt` or environment file
-   Training configuration and exact train/validation/test split
-   Model weights or instructions for obtaining them
-   An inference script with example input and output
-   Evaluation scripts and saved metrics

Do not commit private credentials or large dataset files unless the
repository and dataset terms explicitly permit it.

## Project Team

As listed in the report:

-   **Anshika Shroff**
-   **Ashi Dubey**
-   **Arpit Suman**
-   **Mohan Chauhan**

**Department:** Computer Science, Faculty of Technology, University of
Delhi\
**Guidance:** Dr. Sangeeta Yadav

## References

1.  [Space Debris Detection Dataset ---
    Kaggle](https://www.kaggle.com/datasets/sadianawar/debris-detection-dataset)
2.  The project's accompanying report, *Space Debris Detection Using
    Deep Learning and Exploratory Data Analysis*.

------------------------------------------------------------------------

*This README summarizes the accompanying project report. Confirm dataset
details, experimental settings, and metric calculations against the
source code and logs before using the results in a publication or
benchmark comparison.*
