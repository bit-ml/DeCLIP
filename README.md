# DeCLIP

[![arXiv](https://img.shields.io/badge/-arXiv-B31B1B.svg?style=for-the-badge)](https://doi.org/10.48550/arXiv.2409.08849)

**Official PyTorch Implementation of the Paper:**

> **Ștefan Smeu, Elisabeta Oneață, Dan Oneață**  
> [DeCLIP: Decoding CLIP Representations for Deepfake Localization](https://arxiv.org/pdf/2409.08849)  
> *WACV, 2025*

## Data

To set up your data, follow these steps:

1. **Download the datasets:**
   - **Dolos Dataset:** Follow instructions from [Dolos GitHub repo](https://github.com/bit-ml/dolos)
   - **AutoSplice Dataset:** Follow instructions from [AutoSplice GitHub repo](https://github.com/shanface33/AutoSplice_Dataset)

2. **Organize the data:**

   After downloading, place the datasets in the `datasets` folder to match the following structure:

   ```plaintext
   ├── data/
   ├── datasets/
   │   ├── AutoSplice/
   │   ├── dolos_data/
   │   │   ├── celebahq/
   │   │   │   ├── fake/
   │   │   │   │   ├── lama/
   │   │   │   │   ├── ldm/
   │   │   │   │   ├── pluralistic/
   │   │   │   │   ├── repaint-p2-9k/
   │   │   │   ├── real/
   │   │   ├── ffhq/
   ├── models/
   ├── train.py
   ├── validate.py
   ├── ...

## Installation

Main prerequisites:

* `Python 3.10.14`
* `pytorch=2.2.2 (cuda 11.8)`
* `pytorch-cuda=11.8`
* `torchvision=0.17.2`
* `scikit-learn=1.3.2`
* `pandas==2.1.1`
* `numpy=1.26.4`
* `pillow=10.0.1`
* `seaborn=0.13.0`
* `matplotlib=3.7.1`
* `tensorboardX=2.6.2.2`

## Train

 To train the models mentioned in the article, follow these steps:

 1. **Set up training and validation data paths** in `options/train_options.py` or specify them as arguments when running the training routine.

 2. **Run the training command** using the following template:

 ```bash
 python train.py --name=<experiment_name> --train_dataset=<dataset> --arch=<architecture> --decoder_type=<decoder> --feature_layer=<layer> --fix_backbone --fully_supervised
 ```

 Example commands:

 Train on Repaint-P2:

 ```bash
 python train.py --name=test_repaint --train_dataset=repaint-p2-9k --data_root_path=datasets/dolos_data/celebahq/ --arch=CLIP:ViT-L/14 --decoder_type=conv-20 --feature_layer=layer20 --fix_backbone --fully_supervised
 ```

 Where:

 - `arch` specifies the architecture, such as CLIP:RN50, CLIP:ViT-L/14, CLIP:xceptionnet, or CLIP:ViT-L/14,RN50.
 - `decoder_type` can be linear, attention, conv-4, conv-12, or conv-20.
 - `feature_layer` ranges from layer0 to layer23 for ViTs and from layer1 to layer4 for ResNets.

 Exceptions:

 - For CLIP:xceptionnet, features are always extracted from the 2nd block.
 - For CLIP:ViT-L/14,RN50, the argument value specifies the layer from ViT; for RN50, features are always extracted from layer3.
 - Use `--fully_supervised` for localization tasks. Omit it for image-level detection tasks.

## Pretrained Models
We provide trained models for the networks which rely on ViT and ViT+RN50 backbones listed in the table below.

| Backbone            | Feature Layer | Decoder      | Training Dataset | Download Link                                                                                     |
|---------------------|----------------------------------|--------------|------------------|---------------------------------------------------------------------------------------------------|
| ViT                 | layer20                          | conv-20       | Pluralistic      | [Download](https://storage.cloud.google.com/bitdefender_ml_artifacts/declip/backbone_VIT/ViT_layer20_conv-20_pluralistic.pth)                                           |
| ViT                 | layer20                          | conv-20       | LaMa             | [Download](https://storage.cloud.google.com/bitdefender_ml_artifacts/declip/backbone_VIT/ViT_layer20_conv-20_lama.pth)                                                  |
| ViT                 | layer20                          | conv-20       | RePaint-p2-9k    | [Download](https://storage.cloud.google.com/bitdefender_ml_artifacts/declip/backbone_VIT/ViT_layer20_conv-20_repaint-p2-9k.pth)                                         |
| ViT                 | layer20                          | conv-20       | LDM              | [Download](https://storage.cloud.google.com/bitdefender_ml_artifacts/declip/backbone_VIT/ViT_layer20_conv-20_ldm.pth)                                                   |
| ViT                 | layer20                          | conv-20       | COCO-SD           | [Download](https://storage.cloud.google.com/bitdefender_ml_artifacts/declip/backbone_VIT/ViT_layer20_conv-20_cocosd.pth)                                                |
| ViT+RN50            | layer20+layer3                   | conv-20       | Pluralistic      | [Download](https://storage.cloud.google.com/bitdefender_ml_artifacts/declip/backbone_VIT%2BRN50/ViT_layer20%2BRN50_layer3_conv-20_pluralistic.pth)                               |
| ViT+RN50            | layer20+layer3                   | conv-20       | LaMa             | [Download](https://storage.cloud.google.com/bitdefender_ml_artifacts/declip/backbone_VIT%2BRN50/ViT_layer20%2BRN50_layer3_conv-20_lama.pth)                                      |
| ViT+RN50            | layer20+layer3                   | conv-20       | RePaint-p2-9k    | [Download](https://storage.cloud.google.com/bitdefender_ml_artifacts/declip/backbone_VIT%2BRN50/ViT_layer20%2BRN50_layer3_conv-20_repaint-p2-9k.pth)                             |
| ViT+RN50            | layer20+layer3                   | conv-20       | LDM              | [Download](https://storage.cloud.google.com/bitdefender_ml_artifacts/declip/backbone_VIT%2BRN50/ViT_layer20%2BRN50_layer3_conv-20_ldm.pth)                                       |

Additionally, one can download the checkpoints using **gsutil** from [this GCS bucket](https://console.cloud.google.com/storage/browser/bitdefender_ml_artifacts/declip?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))). The weights are located in **backbone_VIT** and **backbone_VIT+RN50** folders, where each checkpoints follows the naming convention: ```<backbone>_<feature_layer>_<decoder>_<training_dataset>```, **where training_dataset** is lower-cased. For the case of features concatenated from ViT and RN50, a ```+``` charachter joins the 2 backbones and feature layers.


## Evaluation

To evaluate a model, use the following template:

```bash
python validate.py --arch=CLIP:ViT-L/14 --ckpt=path/to/the/saved/mode/checkpoint/model_epoch_best.pth --result_folder=path/to/save/the/results --fully_supervised
```

## License

<p xmlns:cc="http://creativecommons.org/ns#">The code is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0 <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>


This repository also integrates code from the following repositories:
```
@inproceedings{ojha2023fakedetect,
      title={Towards Universal Fake Image Detectors that Generalize Across Generative Models}, 
      author={Ojha, Utkarsh and Li, Yuheng and Lee, Yong Jae},
      booktitle={CVPR},
      year={2023},
}
```
```
@inproceedings{patchforensics,
  title={What makes fake images detectable? Understanding properties that generalize},
  author={Chai, Lucy and Bau, David and Lim, Ser-Nam and Isola, Phillip},
  booktitle={European Conference on Computer Vision},
  year={2020}
 }
```

## Citation

If you find this work useful in your research, please cite it.

```
@InProceedings{DeCLIP,
    author    = {Smeu, Stefan and Oneata, Elisabeta and Oneata, Dan},
    title     = {DeCLIP: Decoding CLIP representations for deepfake localization},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    year      = {2025}
}
```
