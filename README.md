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



```
<dataset>/
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

├── p10
│   ├── p10000032
│   │   ├── s50414267.txt
│   │   ├── s53189527.txt
│   └── ...
├── p11
└── ...

```
