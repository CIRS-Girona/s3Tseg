# ViT-based Semantic Segmentation of the Seafloor in Side-Scan Sonar Data

This repository contains a PyTorch implementation of the ViT-based architecture proposed in *"A convolutional vision transformer for semantic segmentation of side-scan sonar data" published in Ocean Engineering, Volume 86, part 2, 15 October 2023, DOI: [10.1016/j.oceaneng.2023.115647](https://www.sciencedirect.com/science/article/pii/S0029801823020310)* together with implementations of several other ViT-based encoder-decoder architectures for semantic segmentation of the seafloor in side-scan sonar images.

## Getting Started

### Prerequisites

The file [requirements.txt](https://github.com/CIRS-Girona/s3Tseg/blob/main/requirements.txt) contains the necessary Python packages for this project. To install, run:
```
pip install -r requirements.txt
```

All models were trained on an NVIDIA A100 Tensor Core GPU operating on Ubuntu 22.04.2 with Python 3.9.17 and PyTorch 2.0.0+cu120, and evaluated on an NVIDIA Jetson AGX Orin Developer Kit running Jetpack 5.1.1 with Python 3.8.10 and PyTorch 2.0.0+nv23.5.

<!-- The **dataset** used for training is available for download via [this link](https://zenodo.org/records/xxxx). -->

### Training

The file [main.py](https://github.com/CIRS-Girona/s3Tseg/blob/main/main.py) contains the main training loop. It takes the following arguments:
```
--wandb_entity		WandB entity.
--wandb_project		WandB project name.
--wandb_api_key		WandB api key.
--data_dir		Path to training data.
--out_dir		Path to save logs, checkpoints and models.
--encoder		Type of encoder architecture.
--decoder		Type of decoder architecture.
[--config_file]	        Path to configuration file.
[--load_checkpoint]     Path to checkpoint to resume training from.
[--load_weights]        Path to pretrained weights.
[--seed]		Random seed.
[--num_workers]	        Number of data loading workers per GPU.
[--batch_size]	        Number of distinct images loaded per GPU.
```

The arguments in brackets are optional. Further details on WandB specific arguments can be found in [Weights & Biases documentation](https://docs.wandb.ai/guides/track/environment-variables). The default configurations of the implemented architectures can be found in the file [configs/models.py](https://github.com/CIRS-Girona/s3Tseg/blob/main/configs/models.py). Modifications to the configurations of these architectures and to the default training hyperparameters can be optionally done via a yaml file; see [config.yaml](https://github.com/CIRS-Girona/s3Tseg/blob/main/config.yaml) for an example. The file [configs/base.py](https://github.com/CIRS-Girona/s3Tseg/blob/main/configs/base.py), on the other hand, contains all the base configuration parameters.

To train a model comprised of a *sima_tiny* encoder and an *atrous* decoder on a single node with 2 GPUs with user-specified configurations contained in config.yaml, run:
```
torchrun --nproc_per_node=1 --master_port=1234 main.py --wandb_entity <wandb-user-name> --wandb_project <wandb-project-name> --wandb_api_key <wandb-api-key> --data_dir /path/to/sss/dataset --out_dir /path/to/out/dir --config_file /path/to/config.yaml --batch_size 64 --encoder sima_tiny --decoder atrous
```

### Evaluation

The file [eval.py](https://github.com/CIRS-Girona/s3Tseg/blob/main/eval.py) contains qualitative, quantitative and runtime performance evaluation metrics for semantic segmentation. It takes the following arguments:
```
--data_dir        Path to dataset.
--model_path      Path to trained model.
--encoder         Type of encoder architecture.
--decoder         Type of decoder architecture.
[--mode]          Evaluation mode.
[--config_file]   Path to configuration file.
[--cmap_file]     Path to color map file.
[--out_dir]       Path to save evaluation report.
[--device]        Device to compute runtime statistics for.
[--batch_size]    Number of distinct images per batch.
```

The arguments in brackets are optional. Mode can be set to either *quantitative*, *qualitative* or *runtime*. Device can be set to either *cpu* or *cuda*.

To evaluate runtime performance of a model comprised of a *sima_tiny* encoder and an *atrous* decoder with user-specified configurations contained in config.yaml, run:
```
python3 eval.py --encoder sima_tiny --decoder atrous --model_path /path/to/trained/model.pth --data_dir /path/to/sss/test/dataset --out_dir /path/to/out/dir --config_file /path/to/config.yaml --cmap_file /path/to/cmap.csv --batch_size 8 --device cpu --mode runtime
```

## Pretrained Models

Coming soon . . .

## Citation

If you find this repository useful, please consider giving us a star :star:

```
@article{rajani2023s3Tseg,
    title = {A convolutional vision transformer for semantic segmentation of side-scan sonar data},
    author = {Hayat Rajani and Nuno Gracias and Rafael Garcia},
    journal = {Ocean Engineering},
    volume = {286},
    pages = {115647},
    year = {2023},
    issn = {0029-8018},
    doi = {https://doi.org/10.1016/j.oceaneng.2023.115647},
    url = {https://www.sciencedirect.com/science/article/pii/S0029801823020310},
}
```

### Acknowledgement
Our implementation is built upon [[EsViT](https://github.com/microsoft/esvit)] [[Timm](https://github.com/huggingface/pytorch-image-models)]

This work was supported by the DeeperSense project, funded by the European Union’s Horizon 2020 Research and Innovation programme under grant agreement no. [101016958](https://cordis.europa.eu/project/id/101016958).

### Related Projects
[[w-s3Tseg](https://github.com/CIRS-Girona/w-s3Tseg)]
