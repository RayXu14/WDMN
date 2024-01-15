# WDMN
This is the original implementation for wide-and-deep matching network (WDMN). [*Response Ranking with Multi-Types of Deep Interactive Representations in Retrieval-based Dialogues*.]

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
    > 
    > In case you cannot download that data, use this [link](https://www.dropbox.com/scl/fo/awiumbs5ovr8yhtubsm08/h?rlkey=bfumvbusd1xlwouck7uumvwxw&dl=0) (need not to unrar it)
3. Set up the environment (see **Requiments** above)
4. Clone this reposity and train WDMN
    ```bash
    .../WDMN$ cd code
    .../WDMN/code$ git clone https://github.com/RayXu14/WDMN.git
    .../WDMN/code$ cd WDMN
    # (optional) modify run.sh to change the configuration 
    .../WDMN/code/WDMN$ bash run.sh # please make sure the environment is set up properly
    ```

## Results
| Dataset | R_2@1 | R_10@1 | R_10@2 | R_10@5 | MAP | MRR | P@1 |
| ------------------------ | ------- | -------- | -------- | -------- | -------- | -------- | -------- |
| Ubuntu [(Lowe et al., 2015)](https://www.aclweb.org/anthology/W15-4640.pdf) | 0.957 | 0.821 | 0.911 | 0.981 | - | - | - |
| Douban [(Wu et al., 2017)](https://www.aclweb.org/anthology/P17-1046.pdf) | - | 0.301 | 0.460 | 0.799 | 0.594 | 0.644 | 0.490 |
| E-commerce [(Zhang et al., 2018)](https://www.aclweb.org/anthology/C18-1317.pdf) | - | 0.669 | 0.831 | 0.956 | - | - | - |

## Citation
The paper is published at TOIS, 2021 and will be presented at SIGIR, 2022.
