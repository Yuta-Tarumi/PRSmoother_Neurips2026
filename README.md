# Physics-rollout smoother (PR-smoother) 

## Setup using Python virtualenv

The script is tested with Python version 3.12 and the libraries listed in requirementes.txt, 
and a supported GPU.

```
$ python3 -m venv ./venv-pr-smoother
$ . ./venv-pr-smoother/bin/activate
$ pip3 install -r requirements.txt
```

## Run experiments

Lorenz96 (4D): data can be generated on-the-fly.
```
$ python train_script.py --config config/Lorenz96_multimodal_4dim/flow_RealNVP_10step_seed1.yaml
```

For Lorenz96 (40D) and Kolmogorov flow (16,384D) experiments, you need to generate the training and testing data in advance. Code for data generation is stored under codes_datagen. The code assumes that the generated data will be stored under "training_data/" and you can modify this by specifying "root" in the dataset class.

Lorenz96 (40D), linear observation
```
$ python train_script.py --config config/Lorenz96/linear_flow_seed1.yaml
```

Kolmogorov flow (16,384D), full observation
```
$ python train_script.py --config config/Kolmogorov/full_flow_seed1.yaml
```