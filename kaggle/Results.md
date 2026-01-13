- [small](#small)
  - [RoPE](#rope)
    - [seed=50](#seed50)
    - [seed=59](#seed59)
  - [AbsPE](#abspe)
    - [seed=50](#seed50-1)
    - [seed=59](#seed59-1)
  - [None](#none)
    - [seed=50](#seed50-2)
  - [RC](#rc)
    - [seed=50](#seed50-3)
    - [seed=59](#seed59-2)
- [base](#base)
  - [AbsPE](#abspe-1)
    - [seed=59](#seed59-3)
    - [seed=50](#seed50-4)
    - [seed=50](#seed50-5)
    - [seed=28](#seed28)
    - [seed=29](#seed29)
  - [None](#none-1)
    - [seed=59](#seed59-4)


# small
## RoPE
### seed=50
xulin5522/imagenet-small-rope2
Valid Acc: 0.638799965

### seed=59
xulijuan/imagenet-small-rope
Train Loss: 0.8360 | Train Acc: 0.7679 | Valid Acc: 0.5768
Best Accuracy: 0.5812

### sollasi/imagenet-small-rope28
0.586199999

### sollasi/imagenet-small-rope29
0.584999979

## AbsPE
### seed=50
xulin5522/imagenet-small-abs 
Train Loss: 0.1374 | Train Acc: 0.9617 | Valid Acc: 0.6308
Best Accuracy: 0.6360

### seed=59
xulin5522/imagenet-small-abs
Train Loss: 0.7201 | Train Acc: 0.8009 | Valid Acc: 0.5526
Best Accuracy: 0.5562

### pycjn666/imagenet-small-abs29
0.550000012

### pycjn666/imagenet-small-abs28
0.561399996

## None
### seed=50
sinayliu/imagenet-small-none2
Train Loss: 0.2563 | Train Acc: 0.9320 | Valid Acc: 0.5898
Best Accuracy: 0.5918

### du55148/imagenet-small-no29
0.544399977

### du55148/imagenet-small-no28
0.546400011

## RC
### seed=50
sinayliu/imagenet-small-rc
Train Loss: 1.1829 | Aux Loss: 0.0019 | Base Loss: 0.6101 | Train Acc: 0.8232 | Valid Acc: 0.7170
Best Accuracy: 0.7190

### cdong121/cls-small-colrow228
0.709999979

### cdong121/cls-small-colrow229
alpha=600
0.680599988

### b201xiaoli/cls-small-colrow29
alpha=300


### seed=59
wenyangtang/imagenet-small-rc
Train Loss: 0.5993 | Aux Loss: 0.0011 | Base Loss: 0.2777 | Train Acc: 0.9221 | Valid Acc: 0.7200
Best Accuracy: 0.7202


# base
## AbsPE
