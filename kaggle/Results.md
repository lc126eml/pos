- [small](#small)
  - [RoPE](#rope)
    - [seed=50](#seed50)
    - [seed=59](#seed59)
    - [sollasi/imagenet-small-rope28](#sollasiimagenet-small-rope28)
    - [sollasi/imagenet-small-rope29](#sollasiimagenet-small-rope29)
  - [AbsPE](#abspe)
    - [seed=50](#seed50-1)
    - [seed=59](#seed59-1)
    - [pycjn666/imagenet-small-abs29](#pycjn666imagenet-small-abs29)
    - [pycjn666/imagenet-small-abs28](#pycjn666imagenet-small-abs28)
  - [None](#none)
    - [seed=50](#seed50-2)
    - [du55148/imagenet-small-no29](#du55148imagenet-small-no29)
    - [du55148/imagenet-small-no28](#du55148imagenet-small-no28)
  - [RC](#rc)
    - [seed=50](#seed50-3)
    - [cdong121/cls-small-colrow228](#cdong121cls-small-colrow228)
    - [cdong121/cls-small-colrow229](#cdong121cls-small-colrow229)
    - [b201xiaoli/cls-small-colrow229](#b201xiaolicls-small-colrow229)
    - [seed=59](#seed59-2)
- [base](#base)
  - [AbsPE](#abspe-1)
    - [zzr123123/cls-base-abs228](#zzr123123cls-base-abs228)
    - [zzr123123/cls-base-abs229](#zzr123123cls-base-abs229)


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

### b201xiaoli/cls-small-colrow229
alpha=300
Train Loss: 0.9646 | Aux Loss: 0.0019 | Base Loss: 0.3988 | Train Acc: 0.8859 | Valid Acc: 0.7164 | train_time: 743.4s | val_time: 17.1s
Best Accuracy: 0.7188

### seed=59
wenyangtang/imagenet-small-rc
Train Loss: 0.5993 | Aux Loss: 0.0011 | Base Loss: 0.2777 | Train Acc: 0.9221 | Valid Acc: 0.7200
Best Accuracy: 0.7202


# base
## AbsPE

### zzr123123/cls-base-abs228
base_abs_pos_overlap_0_rc_False_alpha_600lr70_s28.csv
Train Loss: 1.6701 | Train Acc: 0.5606 | Valid Acc: 0.4432 | train_time: 2370.5s | val_time: 26.8s
Best Accuracy: 0.4454

### zzr123123/cls-base-abs229
base_abs_pos_overlap_0_rc_False_alpha_600lr70_s29.csv
Train Loss: 1.6479 | Train Acc: 0.5656 | Valid Acc: 0.4390 | train_time: 2372.7s | val_time: 26.8s
Best Accuracy: 0.4396

