# YOLOv8 + DeepSORT Object Tracker

A generic multi-object tracker: **YOLOv8** does per-frame detection and **DeepSORT** links those detections into consistent tracks across frames, drawing a stable bounding box and ID per tracked object as the video plays.

![Python](https://img.shields.io/badge/Python-3.7-3776AB?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=black)
![DeepSORT](https://img.shields.io/badge/DeepSORT-tracking-orange)

Despite the repo's name, the code isn't animal-specific — it tracks whatever classes the loaded YOLO model detects. Swap in an animal-trained model (see [YOLOv8-AnimalTracker2.0](https://github.com/D1abol1cal/YOLOv8-AnimalTracker2.0) for that) or any other `.pt` weights to track a different kind of object.

## How it works

1. `main.py` runs YOLOv8 inference on each frame of an input video.
2. Detections above a confidence threshold are passed to `tracker.py`, a thin wrapper around DeepSORT.
3. DeepSORT's Kalman filter + appearance-feature matching (`deep_sort/`) assigns each detection to an existing track or starts a new one.
4. Each confirmed track is drawn on the frame with a persistent color/ID, and the result is written to an output video.

## Requirements

- Python 3.7
- `pip install -r requirements.txt`

## Setup

1. Download the DeepSORT feature-extraction model ([`mars-small128.pb`](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp)) into `model_data/`.
2. Put your input video in `data/` and a YOLOv8 `.pt` model in `weights/`.
3. Update the filenames at the top of `main.py` (`inputVideo`, `model = YOLO("weights/...")`) to match.
4. Run:
   ```bash
   python main.py
   ```
   The annotated output is written to `output/`.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for detection.
- The DeepSORT implementation vendored in `deep_sort/` is from [computervisiondeveloper/deep_sort](https://github.com/computervisiondeveloper/deep_sort), a fork of the [official DeepSORT](https://github.com/nwojke/deep_sort) repo; see `deep_sort/LICENSE` for its license.
