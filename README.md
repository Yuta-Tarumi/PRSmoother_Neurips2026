# Physics-rollout smoother (PR-smoother)

This codebase reproduces the experiments conducted in our submitted manuscript, "Physics Rollout Smoother (PR-Smoother): Model‑Based Amortized Variational Inference for Physical State-Space Models".

## Setup using Python virtualenv

The script is tested with Python version 3.12 and the libraries listed in requirementes.txt, 
and a supported GPU.

```
$ python3 -m venv ./venv-pr-smoother
$ . ./venv-pr-smoother/bin/activate
$ pip3 install -r requirements.txt
```

## Data	generation

For Lorenz96 (40D) and Kolmogorov flow (16,384D) experiments, you need to generate the training and testing data in advance. Code for data generation is stored under `codes_datagen`. The code assumes that the generated data will be stored under `training_data/` and you can modify this by specifying `root` in the dataset class.

### Lorenz96 (4D)
For Lorenz96 (4D) experiment, data generation prior to the training is not necessary.

### Lorenz96 (40D)
```
# train data: INDEX=0-9999, each file contains 1024 initial conditions, 10,000 files were used in our paper
$ INDEX=0
$ python Lorenz96_datagen.py --index "${INDEX}" --split train

# test data: INDEX=1000000, 1 file was used in our paper
$ INDEX=1000000
$ python Lorenz96_datagen.py --index "${INDEX}" --split test
```

### Kolmogorov flow
```
# train data: INDEX=0-29999, each file contains 96 initial conditions, 30,000 files were used in our paper
$ INDEX=0
$ python Kolmogorov.py --index ${INDEX} --split train

# test data: INDEX=1000000, 1 file was used in our paper
$ INDEX=1000000
$ python Kolmogorov.py --index ${INDEX} --split test

# bias generation
$ python Kolmogorov_bias_generator.py
```

## Run experiments

Lorenz96 (4D)
```
$ python train_script.py --config config/Lorenz96_multimodal_4dim/flow_RealNVP_10step_seed1.yaml
```

Lorenz96 (40D)
```
$ python train_script.py --config config/Lorenz96/linear_flow_seed1.yaml
```

Kolmogorov flow (16,384D)
```
$ python train_script.py --config config/Kolmogorov/full_flow_seed1.yaml
```
