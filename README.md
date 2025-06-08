# DP-Habitat

## DP-Habitat: Bridging the Gap Between Simulation and Reality for Visual Navigation in Dynamic Pedestrian Environments

This repository provides an implementation of DP-Habitat, an extension of the Habitat simulation framework tailored for indoor embodied AI tasks with humanoid agents.

📎 [Project Diagram / Media Link](https://github.com/user-attachments/assets/9693434a-edae-4b10-9224-acae015ade38)

---

## Installation Instructions

This section provides a step-by-step guide for setting up the DP-Habitat environment. The instructions assume prior installation of [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

### 1. Create and Configure the Conda Environment

The following creates a new Conda environment with the required versions of Python and CMake:

```bash
conda create -n hab3_1 python=3.9.19 cmake=3.14.0
conda activate hab3_1
```

### 2. Install Habitat-Sim with Bullet Physics

Install `habitat-sim` using Conda (with Bullet physics and headless rendering), followed by `pybullet`:

```bash
conda install habitat-sim==0.3.1 withbullet headless -c conda-forge -c aihabitat
pip install pybullet
```

### 3. Install DP-Habitat and Habitat-Lab

Clone the repository and install the Habitat-Lab core package:

```bash
git clone https://github.com/qinliangql/DP-Habitat.git
cd habitat-lab/habitat-lab
pip install -e .
```

### 4. Install Habitat-Baselines

To enable learning-based functionalities (e.g., reinforcement learning), install `habitat-baselines`:

```bash
cd habitat-lab/habitat-baseline
pip install -e .
```

---

## Dataset Preparation

### 1. Download Habitat-Compatible Scene Datasets

Refer to the official [Habitat dataset documentation](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md) for details. To download an example MP3D scene:

```bash
python -m habitat_sim.utils.datasets_download --uids mp3d_example_scene --data-path data/
```

Additional assets (such as HM3D, object datasets, etc.) can also be obtained from the same source.

### 2. Download Humanoid Dataset

Download the custom humanoid dataset used in this project from the following link:

* [Baidu Drive Link](https://pan.baidu.com/s/190k7UNGjRKoSy_2WHqxeMQ?pwd=zekq)

### 3. Organize the Data Directory

Ensure that your data directory is structured as follows:

```
data/
├── humanoids/
│   ├── action/
│   └── humanoid_data/
├── objects/
│   ├── example_objects/
│   ├── locobot_merged/
│   └── ycd/
└── scene_datasets/
    └── mp3d_example/
```

---

## Quick Demonstration

To run a quick demonstration of a humanoid agent navigating an indoor environment, open and execute the following notebook:

```bash
quick_demo/human_run_agent_indoor.ipynb
```

This notebook provides a minimal working example to validate the installation and demonstrate the capabilities of DP-Habitat.

---


## Test PPO
DownLoad ppo_gibson_pointnav model from the following link:

* [Baidu Drive Link](https://pan.baidu.com/s/1zbWe2ZCBIz5wfOlawGPy9Q?pwd=4k5e)

Ensure that your data directory is structured as follows:

```
data/
├── datasets/
│   ├── pointnav/
├── humanoids/
│   ├── action/
│   └── humanoid_data/
├── objects/
│   ├── example_objects/
│   ├── locobot_merged/
│   └── ycd/
└── scene_datasets/
    └── mp3d_example/
    └── gibson/

```

---


Test the Simple PPO Model in gibson Datasets:
```bash
cd habitat-lab
bash run.sh
```