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
    - [xulijuan/cls-base-patch129](#xulijuancls-base-patch129)
    - [sollasi/cls-base-alibi-desc-626](#sollasicls-base-alibi-desc-626)
    - [sollasi/cls-base-relpos-desc-626](#sollasicls-base-relpos-desc-626)
  - [20260130](#20260130)
    - [yinmengmeng/cls-base-abs-d-718](#yinmengmengcls-base-abs-d-718)
  - [rc alpha warmup](#rc-alpha-warmup)
    - [djiangjiang/cls-base-colrow-ra600-wsfa1000-16](#djiangjiangcls-base-colrow-ra600-wsfa1000-16)
    - [rrrrmm/cls-base-colrow-ra100-wsfa200-16](#rrrrmmcls-base-colrow-ra100-wsfa200-16)
    - [tianxianglii/cls-base-colrow-ra100-wsfa500-16](#tianxiangliicls-base-colrow-ra100-wsfa500-16)
    - [chencaihonga/cls-base-colrow-ra220-16](#chencaihongacls-base-colrow-ra220-16)
  - [lr ablation](#lr-ablation)
    - [jacksisi/cls-base-none-lr5e-5-53](#jacksisicls-base-none-lr5e-5-53)
    - [jacksisi/cls-base-none-lr2e-4-53](#jacksisicls-base-none-lr2e-4-53)
    - [xulijuan/cls-base-none-lr3e-4-53](#xulijuancls-base-none-lr3e-4-53)
    - [xulijuan/cls-base-none-lr1e-4-53](#xulijuancls-base-none-lr1e-4-53)
    - [liucong12601/cls-base-none-lr8e-5-53](#liucong12601cls-base-none-lr8e-5-53)
    - [liucong12601/cls-base-none-lr7e-5-53](#liucong12601cls-base-none-lr7e-5-53)


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
namespace(pos_type=None, dynamic_img_size=True, model_type='dinov3', use_abs_pos_emb=True, use_rot_pos_emb=False, model_size='base', num_classes=100, patch_size=16, **grad_accum_steps=2, batch_size=64,** img_sizes=[224], val_img_sizes=[160, 176, 192, 208, 224, 256, 272, 288, 320, 336, 352, 368, 384, 400, 416], **lr=0.0003,** lr_aux=1e-05, eta_min=0.0, weight_decay=0.01, epochs=130, overlap=0, pretrained=None, seed=51, use_patch_position_loss=False, use_rc_loss=False, rc_alpha=600.0, warmup_steps_for_aux=1, workers=5, randaugment=False, randaugment_n=2, randaugment_m=3, random_erasing=False, re_prob=0.0, train=True, val=False, ckpt_path=None, lock=True, save_full_ckpt=True, resume_full_ckpt=True, resume_ckpt_path='/kaggle/input/cls-base-abs51/ckpt/last.pth', resume_scheduler=True, resume_optimizer=True, resume_bs=True, composite_lr=True, warmup_steps=3000, clip_value=1.0, log_interval=100, csv_interval=1, show_peak_gpu_mem=True, compile_model=False, total_run_time_hr=12, root_dir='/kaggle/working')
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

### xulijuan/cls-base-patch129
0.550262451	0.958607733	0.717799962	2390.245458	27.61198211	
130	264161	0.155561313	0.000657836

### sollasi/cls-base-alibi-desc-626
0.107592069	0.97344619	0.582799971	2416.06446	29.73661327
Best Accuracy: 0.5864

### sollasi/cls-base-relpos-desc-626
0.052876361	0.987784624	0.613399982	2417.187462	27.6414361
Best Accuracy: 0.6168

## 20260130
namespace(pos_type=None, dynamic_img_size=True, model_type='dinov3', use_abs_pos_emb=True, use_rot_pos_emb=False, model_size='base', num_classes=100, patch_size=16, **grad_accum_steps=2, batch_size=64,** img_sizes=[224], val_img_sizes=[160, 176, 192, 208, 224, 256, 272, 288, 320, 336, 352, 368, 384, 400, 416], **lr=0.00016,** lr_aux=1e-05, eta_min=0.0, weight_decay=0.01, epochs=130, overlap=0, pretrained=None, seed=18, use_patch_position_loss=False, use_rc_loss=False, rc_alpha=600.0, warmup_steps_for_aux=1, workers=5, randaugment=False, randaugment_n=2, randaugment_m=3, random_erasing=False, re_prob=0.0, train=True, val=False, ckpt_path=None, lock=True, save_full_ckpt=True, resume_full_ckpt=True, resume_ckpt_path='/kaggle/input/cls-base-abs-d-618/ckpt/last.pth', resume_scheduler=True, resume_optimizer=True, resume_bs=True, composite_lr=True, warmup_steps=3000, clip_value=1.0, log_interval=500, csv_interval=1, show_peak_gpu_mem=True, compile_model=False, total_run_time_hr=12.0, root_dir='/kaggle/working')

eastwangwei/cls-base-none-d-715
0.29331708	0.927400053	0.627799988	2377.232176	27.13887477	130

robinrainy/cls-base-none-d-716
0.288127244	0.926800013	0.630199969	2360.556972	26.77071929	130

robinrainy/cls-base-none-d-717
0.291850299	0.926946163	0.622999966	2393.38039	27.61225271	130

- kernel_id: maoshuwen0415/cls-base-none-d-718
0.285520136	0.92742312	0.630599976	2382.139256	27.39493513	130

gloden613/cls-base-abs-d-715
0.284928143	0.97000046	0.646799982	2382.102685	27.38780832	130

roseqw/cls-base-abs-d-716
0.27779898	0.929684639	0.636599958	2377.364023	27.34206533	130

eastwangwei/cls-base-abs-d-717
0.280990809	0.929669261	0.65259999	2376.600208	27.06773448	130

### yinmengmeng/cls-base-abs-d-718
0.02142968	0.995469272	0.658399999	2378.702361	27.58827186	130





roseqw/cls-base-colrow-d-715
1.004234672	0.842669249	0.725799978	2397.782822	27.41329002	130

xiaoluoalice/cls-base-colrow-d-716
1.261722803	0.806776941	0.710599959	2381.429861	27.23062944	130

eastwangwei/cls-base-colrow-d-717
1.199402213	0.820361555	0.726399958	2382.091406	27.12340927	130

chencaihonga/cls-base-colrow-d-718
1.056852937	0.845007718	0.727599978	2407.905407	27.6153574	130


## rc alpha warmup
wsfa500 > wsfa1000 > wsfa200 > None
### djiangjiang/cls-base-colrow-ra600-wsfa1000-16
Train Loss: 27.0997 | Aux Loss: 0.0389 | Base Loss: 3.7451 | Train Acc: 0.1360 | Valid Acc: 0.1488 | train_time: 2378.2s | val_time: 26.9s
Best Accuracy: 0.1488
### rrrrmm/cls-base-colrow-ra100-wsfa200-16
Train Loss: 27.4631 | Aux Loss: 0.0394 | Base Loss: 3.8047 | Train Acc: 0.1276 | Valid Acc: 0.1396 | train_time: 2402.4s | val_time: 27.7s
Best Accuracy: 0.1396
### tianxianglii/cls-base-colrow-ra100-wsfa500-16
Train Loss: 26.9665 | Aux Loss: 0.0387 | Base Loss: 3.7362 | Train Acc: 0.1380 | Valid Acc: 0.1496 | train_time: 2401.1s | val_time: 27.7s
Best Accuracy: 0.1496
### chencaihonga/cls-base-colrow-ra220-16
Train Loss: 27.5159 | Aux Loss: 0.0395 | Base Loss: 3.8425 | Train Acc: 0.1208 | Valid Acc: 0.1298 | train_time: 2426.7s | val_time: 27.6s
Best Accuracy: 0.1298

## lr ablation
<!-- grad_accum_steps=2, batch_size=64 -->
### jacksisi/cls-base-none-lr5e-5-53
Train Loss: 2.3781 | Train Acc: 0.3981 | Valid Acc: 0.4054
### jacksisi/cls-base-none-lr2e-4-53
Train Loss: 1.9284 | Train Acc: 0.4996 | Valid Acc: 0.4944
### xulijuan/cls-base-none-lr3e-4-53
Train Loss: 2.0705 | Train Acc: 0.4668 | Valid Acc: 0.4612
### xulijuan/cls-base-none-lr1e-4-53
Train Loss: 1.9759 | Train Acc: 0.4873 | Valid Acc: 0.4874
### liucong12601/cls-base-none-lr8e-5-53
Train Loss: 2.1342 | Train Acc: 0.4529 | Valid Acc: 0.4600
### liucong12601/cls-base-none-lr7e-5-53
Train Loss: 2.1624 | Train Acc: 0.4448 | Valid Acc: 0.4518