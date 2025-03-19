<hr>
<h1 align="center">
  MedTrim <br>
  <sub>MedTrim: Meta-Entity Driven Triplet Mining for Aligning Medical Vision-Language Models</sub>
</h1>
<hr>

Official PyTorch implementation of **MedTrim**

## ⚙️ Installation

This repository has been developed and tested with `CUDA 11.3` and `Python 3.8`. Below commands create a conda environment with required packages. Make sure conda is installed.

```
conda env create --file requirements.txt
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
│   │   │   └── ...
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
