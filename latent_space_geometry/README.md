# Optimal precision for GANs 


## Requirements
A suitable [conda](https://conda.io/) environment named `optiGans` can be created
and activated with:


```
conda env create -f environment.yaml
conda activate optiGans
```


### MNIST
First, train a GAN (with z_dim=128 here):
```
python train.py --name models/mnist_128 --z_dim 128 --n_steps 80001 --dataset mnist
```

Then, evaluate with: 
```
python eval.py --folder_path models/mnist_128 --gen_list gen_80000.pth gen_78000.pth gen_76000.pth gen_74000.pth  gen_72000.pth gen_70000.pth gen_68000.pth gen_66000.pth gen_64000.pth gen_62000.pth --z_dim 128 --dataset mnist
```
### CIFAR-10/CIFAR-100
First, train a GAN (with z_dim=128 here):
```
python train.py --name models/cifar_128 --z_dim 128 --n_steps 100001 --dataset cifar10 --device 1
```
Then, evaluate with:
```
python eval.py --folder_path models/cifar_128 --gen_list gen_100000.pth gen_98000.pth gen_96000.pth gen_94000.pth gen_92000.pth gen_90000.pth gen_88000.pth gen_86000.pth gen_84000.pth gen_82000.pth --z_dim 128 --dataset cifar10
```
For CIFAR100, replace CIFAR10 by CIFAR100: --dataset cifar100.

### CIFAR-10/CIFAR-100 Overparametrization study
First, train a GAN (with z_dim=128 here):
```
python train.py --name models/cifar_128_w256 --z_dim 64 --width 256 --n_steps 100001 --dataset cifar10 --device 0
```
Then, evaluate with:
```
python eval.py --folder_path models/cifar_128_w256 --gen_list gen_100000.pth gen_98000.pth gen_96000.pth gen_94000.pth gen_92000.pth gen_90000.pth gen_88000.pth gen_86000.pth gen_84000.pth gen_82000.pth --z_dim 64 --width 256 --dataset cifar10
```
For CIFAR100, replace CIFAR10 by CIFAR100: --dataset cifar100.

### Synthetic CIFAR-10
1) Train a conditional GAN:
```
python train.py --name models/conditional_cifar10 --z_dim 5 --dataset cifar10 --conditional True --device 0
```
2) Generate a synthetic dataset
```
python generate_synthetic_dataset.py --target_folder data/synthetic_cifar10 --model_path models/conditional_cifar10/gen_100000.pth --z_dim 5 --conditional True --dataset cifar10
```

3) Train standard unsupervised GANs on this dataset:
```
python train.py --name models/synthetic_cifar_128 --z_dim 128 --n_steps 100001 --dataset synthetic_cifar10 --device 0
```

4) Eval the unsupervised GAN
```
python eval.py --folder_path models/synthetic_cifar_128 --gen_list gen_100000.pth gen_98000.pth gen_96000.pth gen_94000.pth gen_92000.pth gen_90000.pth gen_88000.pth gen_86000.pth gen_84000.pth gen_82000.pth --z_dim 128 --dataset synthetic_cifar10
```

```
python eval.py --folder_path models/cifar_128 --gen_list gen_1000.pth gen_6000.pth --z_dim 128 --z_dim 128 --dataset synthetic_cifar10 --device 0
```
