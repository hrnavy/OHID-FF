# OHID-forest-fires
The original image files in the OHID-forest-fires dataset were collected from OHS. The original data images are of size 5056 * 5056, totaling 22 images, located in the tif/fire folder.

The "YOLODataset" folder contains the following:

- **images** subfolder: Contains the remote sensing images after slicing.
- **labels** subfolder: Contains the corresponding label files for these images. The naming of images and labels corresponds one-to-one, with a unified format of `HEM1_20200623235326_0005_L1B_CMOS2_0_8_512_512_0.*`. Here:
  - `HEM1_20200623235326_0005_L1B_CMOS2` indicates the satellite image source of the picture.
  - `0_8_512_512` specifies the position of the slice within the original image.
  - The last digit indicates the presence of a particular category.
  - ALL in the `xywh` format.
- **viz** subfolder: Contains the visualization results of the dataset's labels.
- **classes.txt** file: Contains the category names of the dataset.
- **dataset.yaml** file: Contains the specific paths to the entire dataset. **To use it, you may need to correct the addresses.**

