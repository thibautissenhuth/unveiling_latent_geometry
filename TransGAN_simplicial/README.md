# Simplicial truncation 
Code used for training TransGAN with a simplicial cluster latent space. 

## Guidance
#### Cifar training script
Standard training of TransGAN: 
```
python exps/cifar_train.py
```

Standard training of TransGAN with simplicial cluster latent space: 
```
python exps/cifar_train_simplicial_truncation.py
```

#### Cifar test
Test standard TransGAN on CIFAR-10:
```
python exps/cifar_test.py
```

Test TransGAN with simplicial cluster latent space on CIFAR-10:
```
python exps/cifar_test_simplicial_truncation.py
```

## Acknowledgement
Codebase from [TransGAN](https://github.com/VITA-Group/TransGAN),[AutoGAN](https://github.com/VITA-Group/AutoGAN), [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
```
