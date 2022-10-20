<div align="center">

# Lidar Prod - a tool for the production of Lidar semantic segmentation

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

[![CI/CD](https://github.com/IGNF/lidar-prod/actions/workflows/cicd.yaml/badge.svg?event=push)](https://github.com/IGNF/lidar-prod/actions/workflows/cicd.yaml)
[![Documentation Build](https://github.com/IGNF/lidar-prod/actions/workflows/gh-pages.yaml/badge.svg?event=push)](https://github.com/IGNF/lidar-prod/actions/workflows/gh-pages.yaml)


</div>

## Context
The Lidar HD project ambitions to map France in 3D using 10 pulse/m² aerial Lidar. The data will be openly available, including a semantic segmentation with a minimal number of classes: ground, vegetation, buildings, vehicles, bridges, others.

To produce this classification, geometric rules-based classification are familiar and present advantages such as scalability, high geometric regularity, and predictability. But rules-based algorithm often lack the fine-grain understanding needed for complex Lidar scenes, which results in a need for time-consuming human correction.

Additionnaly, some valuable information exist in 2D public geographical database, but finding a way to leverage it on a point cloud classification is not straightforward considering database incompletness, potential out-of-dateness, and frequent x-y offsets. 

Considering the scale of this task, deep learning is leveraged to as a production tool. A [deep learning library](https://github.com/IGNF/lidar-deep-segmentation) was developed with a focused scope: the multiclass semantic segmentation of large scale, high density aerial Lidar points cloud. Using a classification produced directly by a deep learning model might be tempting, but they usually presents some limitations including unexpected failure modes, inconsistant geometric regularity, noise.

## Content

Lidar-Prod is a production library which aims at augmenting rules-based semantic segmentation algorithms with deep learning neural network predictions (probabilities) and a public building vector database (BDUni). Its main entry-points are:

- `application.py`: The application takes a point cloud and update its Classification dimension based on its deep learning predictions and a public geographic database.
- `optimization.py`: The right balance between automation of decision and error is found via a multi-objective optimization of of the decision thresholds, by means of a simple genetic algorithm.

Our strategy is to fuse together different sources of informations (rules-based classification, deep learning predictions, databases), so that we can ensure a high-quality classification while minimizing the need for human correction. Deep learning probabilities might also be used to highlight area of uncertainty, or to spot elements that were missed by the other approaches.

Right now, the class `building` is the only one that is addressed. The extension to other classes is dependent on the training of multiclass AI model, which requires high quality training datasets that are currently being produced.

> Please refer to the documentation for [installation and usage](https://ignf.github.io/lidar-prod/tutorials/install.html).
    
> Please refer to the documentation to understand the [production process](https://ignf.github.io/lidar-prod/background/production_process.html).

## Version

Decision thresholds for building validation should be optimized for the trained deep learning model which will produce the probabilities.
Current model version:proto151_V1.0_epoch_40_Myria3DV3.0.1.ckpt (accessible at this [Myria3D Production Release](https://github.com/IGNF/myria3d/releases/tag/prod-release-tag) - if up to date).
