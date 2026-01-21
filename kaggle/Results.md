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
    - [sinayliu/cls-base-abs50](#sinayliucls-base-abs50)
    - [sinayliu/cls-base-abs51](#sinayliucls-base-abs51)
    - [denghaimeng/cls-base-abs152](#denghaimengcls-base-abs152)
    - [zhangtingfengztf/cls-base-none350](#zhangtingfengztfcls-base-none350)
    - [zhangtingfengztf/cls-base-none351](#zhangtingfengztfcls-base-none351)
    - [luanjing/cls-base-none152](#luanjingcls-base-none152)
    - [jinzhanbo/cls-base-colrow350](#jinzhanbocls-base-colrow350)
    - [jinzhanbo/cls-base-colrow351](#jinzhanbocls-base-colrow351)
    - [luanjing/cls-base-colrow152](#luanjingcls-base-colrow152)
    - [denghaimeng/cls-base-rope152](#denghaimengcls-base-rope152)
    - [smartchaochao/cls-base-rope151](#smartchaochaocls-base-rope151)
    - [smartchaochao/cls-base-rope150](#smartchaochaocls-base-rope150)


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

### sinayliu/cls-base-abs50
0.237841651	0.936076939	0.579599977	2371.856316	27.13275433

### sinayliu/cls-base-abs51
0.205383256	0.94479233	0.586199999	2400.904654	27.77643204

### denghaimeng/cls-base-abs152
0.133362517	0.969623089	0.566599965	2373.718566

 ### zhangtingfengztf/cls-base-none350
0.254755884	0.931753874	0.557399988	2378.142009	27.15587854

 ### zhangtingfengztf/cls-base-none351
 0.288951635	0.921869278	0.546400011	2366.575129	26.79268241

### luanjing/cls-base-none152
0.125011474	0.971784651	0.556400001	2372.97083	29.72930169

### jinzhanbo/cls-base-colrow350
0.728125691	0.914353848	0.724599957	2391.610577	27.32547092

### jinzhanbo/cls-base-colrow351
0.697208643	0.928892314	0.720999956	2411.936993	27.77830696

### luanjing/cls-base-colrow152
0.824765563	0.920300007	0.709199965	2377.724379	29.48022437


### denghaimeng/cls-base-rope152
0.09755639	0.976215422	0.620400012	2596.61759	32.16709471

### smartchaochao/cls-base-rope151
0.115926519	0.971815407	0.626999974	2599.581197	32.18054485

### smartchaochao/cls-base-rope150
0.110057667	0.973015428	0.624000013	2621.189191	32.41515899

