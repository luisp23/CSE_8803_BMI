# continual_diffuser
CSE 8803 Final Project


## Installation
```bash
conda env create -f environment.yml
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Set these your `~/.bashrc`: 

```bash 
export MUJOCO_PY_MUJOCO_PATH=~/scratch/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hice1/lpimentel3/scratch/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
pip install "Cython<3" # should this be in env?
```

- `No such file or directory: 'patchelf' on mujoco-py installation` error fix
```bash
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3

export CPATH=$CONDA_PREFIX/include
pip install patchelf
```

Get dataset: https://drive.google.com/drive/folders/1i9wOBM9zRirWbJ4tIKBqXsIH6fqsWac5


### Installing Clean Diffuser

```bash 
git submodule init
git submodule update
git clone https://github.com/Farama-Foundation/D4RL.git

```




Change the dependencies in the `pyproject.toml` to the following: 
```python 
dependencies = [
    "einops",
    "zarr>=2.16.1",
    "av>=12.2.0",
    "dill>=0.3.8",
    "dm_control>=1.0.3",
    "numba<0.60.0",
    "six",
    "hydra-core",
    "zarr<2.17",
    "wandb",
    "dill",
    "av",
    "pygame",
    "pymunk",
    "shapely<2.0.0",
    "scikit-image<0.23.0",
    "opencv-python",
    "imagecodecs",
]
```
Then install package 

```bash 
cd CleanDiffuser 
pip install -e . 
```

fix this dependency issue before running
```bash 
pip install "dm_control<=1.0.20" "mujoco<=3.1.6"
pip install gym==0.13
pip install cloudpickle==1.3.0
```

Running tutorial: 

training
```bash 
python CuGRO/clean_diffuser_tutorial.py --mode=training
```

evaluation after traning: 
```bash 
python CuGRO/clean_diffuser_tutorial.py --mode=evaluation
```




## Running CuGRO

Training behavior and generator: 

```bash 
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  main-gene.py --env "swimmer_dir" --data_mode "gene" --actor_type "large" --diffusion_steps 100
```

`--master_port=29501` : add argument to run multiple at the same time (1 GPU/process)


Training critic and evaluation: 
```python 
python critic.py --env "cheetah_vel" --data_mode "gene" --actor_type "large" --diffusion_steps 100 --gpu 0
```

## Running Continual Diffuser

Training: 
```bash 
python train_continual_diffuser.py --env cheetah_vel --data_mode continual_diffuser --trajectory_horizon 4
```

Evaluation:
```bash 
python eval_continual_diffuser.py --env cheetah_vel --data_mode continual_diffuser --actor_load_epoch 600 --trajectory_horizon 4 --ending_task 3
```

## Debugging 
- https://github.com/openai/mujoco-py/issues/652
- https://github.com/openai/mujoco-py/issues/627



