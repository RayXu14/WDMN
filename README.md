# WDMN
This is the orinal implementation for wide-and-deep matching network (WDMN). [*Response Ranking with Multi-Types of Deep Interactive Representations in Retrieval-based Dialogues*.]

## Requirements
### Option 1: using container image
[tensorflow/tensorflow:1.15.5-gpu-py3](https://hub.docker.com/layers/tensorflow/tensorflow/1.15.5-gpu-py3/images/sha256-7ad742cbb2c77a40d0996ec08345dff54fe25f39428486e5f41c69db04b6d17b?context=explore)
### Option 2: building Python environment on your own
* python==3.7.6
* numpy==1.18.1
* tensorflow-gpu==1.15.0

## Usage
1. Build directory structure
    ```bash
    ...$ mkdir WDMN && cd WDMN
    .../WDMN$ mkdir code data
    ```
2. [Download data](https://drive.google.com/drive/folders/1pJKIppcbjuTZxbTc8ye5mfnC2ygR2xTo) and unzip it
    ```bash
    .../WDMN$ unrar x dataset.rar data
    ```
    > Thank [ chunyuanY ](https://github.com/chunyuanY) for providing the data.
3. Set up the environment (see **Requiments** above)
4. Clone this reposity and train WDMN
    ```bash
    .../WDMN$ cd code
    .../WDMN/code$ git clone https://github.com/RayXu14/WDMN.git
    .../WDMN/code$ cd WDMN
    # (optional) modify run.sh to change the configuration 
    .../WDMN/code/WDMN$ bash run.sh # please make sure the environment is set up properly
    ```

## Citation
The paper will be published on TOIS (the ACM Transactions on Information Systems) soon.
