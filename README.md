# SAda-Net
### A Self-Supervised Adaptive Stereo Estimation CNN For Remote Sensing Image Data
Dominik Hirner, Friedrich Fraundorfer
![Network](./docs/Network_header.png)

A pytorch implementation of our completely self-supervised stereo method for remote sensing image data.
This method has been accepted and will be published at the **ICPR 2024** conference. If you use our work please cite our paper (link later).

Our training routine does not need any annotated ground truth data and is therefore well suited for remote sensing applications, where detailed annotated ground truth data is often missing. We achieve this by introducing a training scheme based on a pseudo ground-truth. This is done by using the initial noisy disparity map as a starting point for training. For robustness noisy/wrong predictions are removed using the left-right consistency check. The so created sparse disparity map is then used as the pseudo ground-truth for training. After each training step this pseudo ground-truth is updated. To track convergence the number of all removed inconsistent point is tracked. 

Our training loop visualized: 
![Train Loop](./docs/Training_Loop.png)

Tracking of inconsistent points visualized: 
![Incons Points](./docs/Tracking_Incons.png)

The whole project is in python 3 and pytorch 1.8.0 and Cuda 12.4.

This repository contains

- jupyter notebooks for training and inference of disparity via a stereo-pair
- link to our s2p fork with modifications for deep learning matching
- python3.6 code for training and inference
- trained weights for the DFC-2019 contest

## Usage
We use a trainable guided filter for the cost-volume (see [project](http://wuhuikai.me/DeepGuidedFilterProject/)). This can be installed via pip.

 ```pip install guided-filter-pytorch```

We use OpenCV for multiple image manipulations. 
```pip install opencv-python```

For the geotiff manipulation we use gdal. On Ubuntu install with:
```sudo apt install libgeotiff-dev libgdal-dev```

If you have problems installing gdal try following the install instruction from the s2p software (see [s2p](https://github.com/centreborelli/s2p))

### Training 
If you want to train our method on your own data, we provide two different scripts in order to do this.
If you want to use jupyter notebook (or google colab) for training, you can check out and run *SAda-Net_train.ipynb* in the root of this repository. 
If you want a pure python3 implementation, we also provide a *sadanet_train.py* script in the root of this repository.
Usage: ```python sadanet_train.py```. 
We provide a dataloader function for Middlebury, *loadMB*, KITTI2012, *loadKitti2012* and for georeferenced panchromatic tiles, such as ones created by s2p, *loadJack*. For the georeferenced case, the script assumes that all left and corresponding right tiles have the same name but be stored in different folders.
For example:
```./left_tiles_folder/tile1.tif```
```./right_tiles_folder/tile1.tif```

### Inference 
#### Example on Middlebury
## Examples
