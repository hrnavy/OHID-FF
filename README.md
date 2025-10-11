# OHID-FF

The OHID-FF dataset contains remote sensing images collected from OHS. The original images are high-resolution (5056 × 5056 pixels) and are stored in the `tif/fire` folder. There are a total of 22 images in this folder.

## Dataset Structure

The `YOLODataset` folder is organized as follows:

- **images/**  
  Contains the sliced remote sensing images derived from the originals.

- **labels/**  
  Contains label files for each sliced image. Each label file matches its corresponding image by name. The file naming convention is:
  ```
  HEM1_20200623235326_0005_L1B_CMOS2_0_8_512_512_1.txt
  ```
  - `HEM1_20200623235326_0005_L1B_CMOS2`: Indicates the satellite image source.
  - `0_8_512_512`: Specifies the position of the slice within the original image.
  - The last digit (`1` in the example above) indicates the presence of a particular category.
  - All label coordinates are in `xywh` format.

- **viz/**  
  Contains visualizations of the dataset’s label annotations.

- **classes.txt**  
  Lists the category names used in the dataset.

- **dataset.yaml**  
  Provides the paths to all dataset components.  
  **Note:** You may need to update the file paths in this YAML file to match your local setup.

## Dataset Splitting

- **Split_dataset.ipynb**  
  This Jupyter Notebook is used to split the dataset for training, validation, and testing.  
  Use this notebook to generate your desired dataset splits before training your models.
- **file_list.csv**  
  Result from Split_dataset.ipynb.

## Summary

- Original images: `tif/fire/` (22 files, 5056 × 5056 pixels each)
- Sliced images and labels: `YOLODataset/images/`, `YOLODataset/labels/`
- Visualization: `YOLODataset/viz/`
- Category list: `YOLODataset/classes.txt`
- Dataset configuration: `YOLODataset/dataset.yaml` (update paths as needed)
- Dataset splitting: `Split_dataset.ipynb` (use for preparing train/val/test splits)
