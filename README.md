<hr>
<h1 align="center">
  MedTrim <br>
  <sub>Meta-Entity Driven Triplet Mining for Aligning Medical Vision-Language Models</sub>
</h1>

Official PyTorch implementation of **MedTrim**

## ⚙️ Installation

This repository has been developed and tested with `CUDA 11.7` and `Python 3.8`. Below commands create a conda environment with required packages. Make sure conda is installed.

```
conda env create --file requirements.yaml
conda activate medtrim
```

## 🗂️ Prepare dataset

The dataset is divided into two main sections: one for images and one for text reports. Each section is organized hierarchically to reflect patient and study (subject) information.

```
<image dataset>/
├── p10
│   ├── p10000032
│   │   ├── s50414267
│   │   │   ├── 174413ec-4ec4c1f7-34ea26b7-c5f994f8-79ef1962.jpg
│   │   ├── s53189527
│   │   │   ├── e084de3b-be89b11e-20fe3f9f-9c8d8dfe-4cfd202c.jpg
│   │   └── ...
├── p11
└── ...

<text dataset>/
├── p10
│   ├── p10000032
│   │   ├── s50414267.txt
│   │   ├── s53189527.txt
│   └── ...
├── p11
└── ...

```

Run the following command to start OBER Algorithm:

```
python data/run_ober.py --input /path/to/your/input.csv.gz --output /path/to/your/output.csv.gz
```

Run the following command to start Triplet Generation Algorithm:

```
python data/run_triplet_generation.py --input /path/to/input.pkl --output /path/to/output.csv.xz \
    --threshold 0.25 --semi_hard_prob 1.0 --big_batch_size 512 --mini_batch_size 32 --total_iter 40000
```

## 🏃 Training

Run the following command to start training:

```
python project_run/main.py
```

# Argument Descriptions

This document provides descriptions for the configurable parameters in the `config.yaml` file.

## **Arguments Table**

| Argument                     | Description                                                                                                                       |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `--config`                   | Path to the configuration file.                                                                                                  |
| `--data.img_df`              | Path to the image dataset CSV file.                                                                                              |
| `--data.text_df`             | Path to the text dataset CSV file.                                                                                               |
| `--data.triplet_csv`         | Path to the triplet CSV file containing triplet samples.                                                                        |
| `--data.model_save_path`     | Directory path where trained models will be saved.                                                                              |
| `--training.batch_size`      | Batch size for training.                                                                                                        |
| `--training.num_workers`     | Number of workers for data loading.                                                                                             |
| `--training.learning_rate`   | Learning rate for optimizers.                                                                                                   |
| `--training.weight_decay`    | Weight decay (L2 regularization) for optimizer.                                                                                |
| `--training.num_epochs`      | Number of epochs to train the model.                                                                                            |
| `--training.save_freq`       | Frequency (in epochs) to save model checkpoints.                                                                                |
| `--training.device`          | Device for training (e.g., `cuda:0`, `cuda:1`, `cpu`).                                                                          |
| `--training.random_seed`     | Random seed for reproducibility.                                                                                                |
| `--models.text_model`        | Pretrained transformer model for text encoding (default: `"emilyalsentzer/Bio_ClinicalBERT"`).                                 |
| `--models.img_model`         | Pretrained vision model for image encoding (default: `"google/vit-base-patch16-224"`).                                         |
| `--models.margin`            | Margin value for triplet loss.                                                                                                  |

---

## ✒️ Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```
@article{,
  title={MedTrim: Meta-Entity Driven Triplet Mining for Aligning Medical Vision-Language Models}, 
  author={},
  year={},
  journal={}
}
```

<hr>

Copyright © 2025, ICON Lab.
