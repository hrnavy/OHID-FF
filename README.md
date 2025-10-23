# OHID-FF

The OHID-FF dataset contains high-resolution remote sensing images (5056 × 5056 px) collected from OHS. This repository contains the original imagery and a prepared YOLO-style sliced dataset used for object detection and binary fire/non-fire classification experiments.

## Repository layout

- tif/fire/  
  Original high-resolution TIFF images (22 files, 5056 × 5056 px each).

- YOLODataset/  
  - images/ — Sliced 512×512 images used for training.  
  - labels/ — YOLO-format label files (xywh, normalized) matching each sliced image.  
  - viz/ — Visualizations of labels overlaid on slices.  
  - classes.txt — Category names (one per line).  
  - dataset.yaml — Dataset configuration providing paths to images and labels. Update paths as needed for your environment.

- train val scripts/  
  Classification experiments and training scripts for fire/non-fire models (see the folder README for usage details).

- Split_dataset.ipynb  
  Notebook to create train/val/test splits and produce file_list.csv.

## Label file naming convention

Example filename:
HEM1_20200623235326_0005_L1B_CMOS2_0_8_512_512_1.txt

- `HEM1_20200623235326_0005_L1B_CMOS2` — original source image identifier.  
- `0_8_512_512` — position and size of the slice inside the original image (x_y_w_h).  
- Final digit indicates whether the slice contains the target category (1 = contains).  
- Label coordinates are YOLO `xywh` normalized format (center_x center_y width height).

## Quick start

1. Install dependencies (for the classification scripts):
```bash
pip install -r "train val scripts/requirements.txt"
```

2. Prepare the YOLODataset structure (if you need to rebuild it):
```bash
python "train val scripts/prepare_data.py"
```

3. Check and update dataset paths in:
- YOLODataset/dataset.yaml
- train val scripts/config.py (if used by your training scripts)

4. Run training / experiments:
- For object detection with your chosen YOLO implementation, point the trainer at `YOLODataset/dataset.yaml` and `YOLODataset/classes.txt`.
- For binary classification experiments with the included scripts:
```bash
python "train val scripts/main.py"
```

Results and logs from training runs are saved under `results/` (see the scripts folder README for details).

## Dataset splitting

- Use `Split_dataset.ipynb` to generate `file_list.csv` and produce train/val/test splits.
- The notebook uses stratified sampling to preserve class balance; update parameters in the notebook if you need a different split ratio.

## Dataset summary (as provided)

- Original images: 22 TIFFs at 5056 × 5056 px  
- Sliced images: 512 × 512 px in `YOLODataset/images/`  
- Labels: YOLO-format labels in `YOLODataset/labels/`  
- Classes file: `YOLODataset/classes.txt`

(From classification experiments folder: dataset size = 1,197 images (512×512), class distribution: 647 fire / 550 non-fire.)

## Contributing

Contributions, issues, and feature requests are welcome. If you add scripts or tools that change dataset paths or formats, please update `YOLODataset/dataset.yaml` and this README accordingly.

## License

Add a LICENSE file to the repository to specify licensing terms. If no license file exists in the repo, default repository copyright applies.

## Contact

Maintainer: hrnavy

## Changes in this update

- Clarified repository layout and dataset paths.  
- Added quick-start instructions for both YOLO-style detection use and the binary classification scripts.  
- Emphasized the need to update `dataset.yaml` to match local paths.  
- Pointed to the `train val scripts/` README for model-specific commands and dependencies.
