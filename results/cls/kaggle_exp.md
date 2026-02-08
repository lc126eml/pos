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
    - [asdsad0000/cls-base-rope-d-815](#asdsad0000cls-base-rope-d-815)
    - [ampere888/cls-base-rope-d-816](#ampere888cls-base-rope-d-816)
    - [ampere888/cls-base-rope-d-817](#ampere888cls-base-rope-d-817)
    - [asdsad0000/cls-base-rope-d-818](#asdsad0000cls-base-rope-d-818)
    - [roseqw/cls-base-colrow-d-715](#roseqwcls-base-colrow-d-715)
    - [xiaoluoalice/cls-base-colrow-d-716](#xiaoluoalicecls-base-colrow-d-716)
    - [eastwangwei/cls-base-colrow-d-717](#eastwangweicls-base-colrow-d-717)
    - [chencaihonga/cls-base-colrow-d-718](#chencaihongacls-base-colrow-d-718)
  - [compare methods](#compare-methods)
  - [mres](#mres)
  - [small](#small-1)
  - [rc alpha warmup](#rc-alpha-warmup)
    - [djiangjiang/cls-base-colrow-ra600-wsfa1000-16](#djiangjiangcls-base-colrow-ra600-wsfa1000-16)
    - [djiangjiang/cls-base-colrow-ra600-wsfa1000-116](#djiangjiangcls-base-colrow-ra600-wsfa1000-116)
    - [rrrrmm/cls-base-colrow-ra100-wsfa200-16](#rrrrmmcls-base-colrow-ra100-wsfa200-16)
    - [rrrrmm/cls-base-colrow-ra100-wsfa200-116](#rrrrmmcls-base-colrow-ra100-wsfa200-116)
    - [tianxianglii/cls-base-colrow-ra100-wsfa500-16](#tianxiangliicls-base-colrow-ra100-wsfa500-16)
    - [tianxianglii/cls-base-colrow-ra100-wsfa500-116](#tianxiangliicls-base-colrow-ra100-wsfa500-116)
    - [chencaihonga/cls-base-colrow-ra220-16](#chencaihongacls-base-colrow-ra220-16)
    - [chencaihonga/cls-base-colrow-ra220-116](#chencaihongacls-base-colrow-ra220-116)
  - [ra works](#ra-works)
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
namespace(pos_type=None, dynamic_img_size=True, model_type='dinov3', use_abs_pos_emb=True, use_rot_pos_emb=False, model_size='base', num_classes=100, patch_size=16, **grad_accum_steps=2, batch_size=64,** img_sizes=[224], val_img_sizes=[160, 176, 192, 208, 224, 256, 272, 288, 320, 336, 352, 368, 384, 400, 416], **lr=0.0003,** lr_aux=1e-05, eta_min=0.0, weight_decay=0.01, epochs=130, overlap=0, pretrained=None, seed=51, use_patch_position_loss=False, use_rc_loss=False, rc_alpha=600.0, warmup_steps_for_aux=600, workers=5, randaugment=False, randaugment_n=2, randaugment_m=3, random_erasing=False, re_prob=0.0, train=True, val=False, ckpt_path=None, lock=True, save_full_ckpt=True, resume_full_ckpt=True, resume_ckpt_path='/kaggle/input/cls-base-abs51/ckpt/last.pth', resume_scheduler=True, resume_optimizer=True, resume_bs=True, composite_lr=True, warmup_steps=3000, clip_value=1.0, log_interval=100, csv_interval=1, show_peak_gpu_mem=True, compile_model=False, total_run_time_hr=12, root_dir='/kaggle/working')
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
namespace(pos_type=None, dynamic_img_size=True, model_type='dinov3', use_abs_pos_emb=True, use_rot_pos_emb=False, model_size='base', num_classes=100, patch_size=16, **grad_accum_steps=2, batch_size=64,** img_sizes=[224], val_img_sizes=[160, 176, 192, 208, 224, 256, 272, 288, 320, 336, 352, 368, 384, 400, 416], **lr=0.00016,** lr_aux=1e-05, eta_min=0.0, weight_decay=0.01, epochs=130, overlap=0, pretrained=None, seed=18, use_patch_position_loss=False, use_rc_loss=False, rc_alpha=600.0, warmup_steps_for_aux=600, workers=5, randaugment=False, randaugment_n=2, randaugment_m=3, random_erasing=False, re_prob=0.0, train=True, val=False, ckpt_path=None, lock=True, save_full_ckpt=True, resume_full_ckpt=True, resume_ckpt_path='/kaggle/input/cls-base-abs-d-618/ckpt/last.pth', resume_scheduler=True, resume_optimizer=True, resume_bs=True, composite_lr=True, warmup_steps=3000, clip_value=1.0, log_interval=500, csv_interval=1, show_peak_gpu_mem=True, compile_model=False, total_run_time_hr=12.0, root_dir='/kaggle/working')

train_loss	train_acc	valid_acc	train_time	val_time	epoch
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



### asdsad0000/cls-base-rope-d-815
0.264511794	0.928061545	0.681999981	2605.115828	32.36702776	130

### ampere888/cls-base-rope-d-816
0.265429288	0.928661585	0.674799979	2601.882329	32.18635082	130

### ampere888/cls-base-rope-d-817
0.261341631	0.928861558	0.684599996	2582.073358	31.79768419	130

### asdsad0000/cls-base-rope-d-818
0.257901967	0.931200027	0.682399988	2614.811826	32.7463088	130

### roseqw/cls-base-colrow-d-715
namespace(pos_type=None, dynamic_img_size=True, model_type='dinov3', use_abs_pos_emb=False, use_rot_pos_emb=False, model_size='base', num_classes=100, patch_size=16, grad_accum_steps=2, batch_size=64, img_sizes=[224], val_img_sizes=[160, 176, 192, 208, 224, 256, 272, 288, 320, 336, 352, 368, 384, 400, 416], **lr=0.00016,** lr_aux=1e-05, eta_min=0.0, weight_decay=0.01, epochs=130, overlap=0, pretrained=None, seed=15, use_patch_position_loss=False, use_rc_loss=True, rc_alpha=600.0, warmup_steps_for_aux=1, alpha_min=10, workers=5, randaugment=False, randaugment_n=2, randaugment_m=3, random_erasing=False, re_prob=0.0, train=True, val=False, ckpt_path=None, lock=True, save_full_ckpt=True, resume_full_ckpt=True, resume_ckpt_path='/kaggle/input/cls-base-colrow-d-615/ckpt/last.pth', resume_scheduler=True, resume_optimizer=True, resume_bs=True, composite_lr=True, warmup_steps=3000, clip_value=1.0, log_interval=500, csv_interval=1, show_peak_gpu_mem=True, compile_model=False, total_run_time_hr=12.0, root_dir='/kaggle/working')
1.004234672	0.842669249	0.725799978	2397.782822	27.41329002	130

### xiaoluoalice/cls-base-colrow-d-716
1.261722803	0.806776941	0.710599959	2381.429861	27.23062944	130

### eastwangwei/cls-base-colrow-d-717
1.199402213	0.820361555	0.726399958	2382.091406	27.12340927	130

### chencaihonga/cls-base-colrow-d-718
1.056852937	0.845007718	0.727599978	2407.905407	27.6153574	130

## compare methods

- kernel_id: xulin5522/cls-base-alibi-gpu-716
0.521085918	0.854723096	0.642399967	2425.157477	30.01611304	130

- kernel_id: sinayliu/cls-base-alibi-gpu-717
0.542885542	0.850315392	0.637399971	2399.584164	29.23903489	130

- kernel_id: yangyangchengcheng/cls-base-alibi-gpu-718
0.565398753	0.843730807	0.635599971	2404.211515	29.52222633	130

- kernel_id: xulin5522/cls-base-relpos-gpu-716
0.455137223	0.874123096	0.667599976	2414.408037	28.01959133	130

- kernel_id: sinayliu/cls-base-relpos-gpu-717
0.481518537	0.867515385	0.65259999	2418.578123	27.86309576	130

- kernel_id: qqmail4092/cls-base-relpos-gpu-718
0.458218724	0.874876976	0.654399991	2426.265976	27.96287632	130

- kernel_id: sollasi/cls-base-patch-ra600-wsfa600-716
2.73837328	0.620615423	0.600799978	2370.845325	27.06809878	130


## mres
- kernel_id: starysinger/cls-base-none-mres-716
0.442209989	0.87975198	0.635199964	2384.479861	27.99359727	130

- kernel_id: zjl001/cls-base-none-mres-717
0.496297657	0.864074826	0.632999957	2385.446776	28.10336423	130

- kernel_id: ttzyty/cls-base-abs-mres-716
0.399724305	0.888936758	0.661599994	2377.711279	27.79076624	130

- kernel_id: sollasi/cls-base-abs-mres-717
0.42206341	0.883921325	0.647199988	2399.078232	28.22744656	130

- kernel_id: xx03071425/cls-base-rope-mres-816
0.43903172	0.878051996	0.669399977	2607.728029	32.36809421	130

- kernel_id: xulijuan/cls-base-rope-mres-817
0.463641673	0.870751858	0.680599988	2582.120093	31.99543047	130

- kernel_id: qqmail4092/cls-base-rope-mres-818
0.468748182	0.869113386	0.676199973	2587.12902	32.02012062	130

- kernel_id: maoshuwen0415/cls-base-colrow-mres-716
namespace(pos_type=None, dynamic_img_size=True, model_type='dinov3', use_abs_pos_emb=False, use_rot_pos_emb=False, model_size='base', num_classes=100, patch_size=16, grad_accum_steps=2, batch_size=64, img_sizes=[224, 192, 288], val_img_sizes=[160, 176, 192, 208, 224, 256, 272, 288, 320, 336, 352, 368, 384, 400, 416], **lr=5e-05**, lr_aux=1e-05, eta_min=0.0, weight_decay=0.01, epochs=130, overlap=0, pretrained=None, seed=16, use_patch_position_loss=False, use_rc_loss=True, rc_alpha=600, warmup_steps_for_aux=600, alpha_min=10, workers=5, randaugment=False, randaugment_n=2, randaugment_m=3, random_erasing=False, re_prob=0.0, train=True, val=False, ckpt_path=None, lock=True, save_full_ckpt=True, resume_full_ckpt=True, resume_ckpt_path='/kaggle/input/cls-base-colrow-mres-616/ckpt/last.pth', resume_scheduler=True, resume_optimizer=True, resume_bs=True, composite_lr=True, warmup_steps=3000, clip_value=1.0, log_interval=500, csv_interval=1, show_peak_gpu_mem=True, compile_model=False, total_run_time_hr=12.0, root_dir='/kaggle/working')
2.598177195	0.595278382	0.577399969	2398.979801	28.03303885	130

- kernel_id: zhuangminghui/cls-base-colrow-mres-717
2.479523897	0.602716982	0.602199972	2387.708254	27.6904273	130

- kernel_id: qqmail4092/cls-base-colrow-mres-718
2.360729933	0.618324935	0.611000001	2381.52163	27.72110963	130


- kernel_id: zjl001/cls-base-colrow-mres-719
2.111586571	0.637202144	0.628399968	2405.450285	28.15464258	130

- kernel_id: sollasi/cls-base-patch-mres-716
2.35313201	0.643148363	0.622799993	2401.536635	28.96081257	130

## small
  - kernel_id: qcrqcrqcr/cls-small-none-gpu-216
Train Loss: 0.2397 | Train Acc: 0.9355 | Valid Acc: 0.6298 | train_time: 733.8s | val_time: 16.0s
Best Accuracy: 0.6334
  - kernel_id: qcrqcrqcr/cls-small-none-gpu-217
Train Loss: 0.2295 | Train Acc: 0.9376 | Valid Acc: 0.6282 | train_time: 733.4s | val_time: 16.3s2026-02-07 05:11:46,333 - INFO - Best Accuracy: 0.6294
Best Accuracy: 0.6294
  - kernel_id: dingziheng11/cls-small-abs-gpu-216
Train Loss: 0.2074 | Train Acc: 0.9430 | Valid Acc: 0.6456 | train_time: 734.3s | val_time: 16.2s
Best Accuracy: 0.6476
  - kernel_id: dingziheng11/cls-small-abs-gpu-217
Train Loss: 0.2048 | Train Acc: 0.9435 | Valid Acc: 0.6534 | train_time: 731.7s | val_time: 16.1s
2026-02-07 05:10:57,217 - INFO - Best Accuracy: 0.6548
  - kernel_id: zhoushuqing/cls-small-rope-gpu-217
Train Loss: 0.2140 | Train Acc: 0.9404 | Valid Acc: 0.6830 | train_time: 838.8s | val_time: 16.2s
Best Accuracy: 0.6850
  - kernel_id: zhoushuqing/cls-small-rope-gpu-216
Train Loss: 0.2303 | Train Acc: 0.9365 | Valid Acc: 0.6670 | train_time: 839.6s | val_time: 16.6s
2026-02-07 09:12:55,154 - INFO - Best Accuracy: 0.6722
  - kernel_id: guyuefangyuan6666/cls-small-colrow-gpu-216
  Train Loss: 1.8797 | Aux Loss: 0.0017 | Base Loss: 0.8887 | Train Acc: 0.7487 | Valid Acc: 0.6902 | train_time: 740.2s | val_time: 17.7s
1.879690289	0.748746157	0.690199971	740.1846721	17.73873115	130	264161
Best Accuracy: 0.6920
  - kernel_id: guyuefangyuan6666/cls-small-colrow-gpu-217
Train Loss: 1.4761 | Aux Loss: 0.0011 | Base Loss: 0.7887 | Train Acc: 0.7745 | Valid Acc: 0.7042 | train_time: 738.8s | val_time: 15.9s
2026-02-07 05:33:37,596 - INFO - Best Accuracy: 0.7048

## rc alpha warmup
16: wsfa500 > wsfa1000 > wsfa200 > None
116: wsfa1000 > wsfa500 > None > wsfa200
### djiangjiang/cls-base-colrow-ra600-wsfa1000-16
Train Loss: 27.0997 | Aux Loss: 0.0389 | Base Loss: 3.7451 | Train Acc: 0.1360 | Valid Acc: 0.1488 | train_time: 2378.2s | val_time: 26.9s
Best Accuracy: 0.1488
### djiangjiang/cls-base-colrow-ra600-wsfa1000-116
Train Loss: 21.3115 | Aux Loss: 0.0308 | Base Loss: 2.8063 | Train Acc: 0.3074 | Valid Acc: 0.3090 | train_time: 2379.1s | val_time: 27.1s
Best Accuracy: 0.3090

### rrrrmm/cls-base-colrow-ra100-wsfa200-16
Train Loss: 27.4631 | Aux Loss: 0.0394 | Base Loss: 3.8047 | Train Acc: 0.1276 | Valid Acc: 0.1396 | train_time: 2402.4s | val_time: 27.7s
Best Accuracy: 0.1396
### rrrrmm/cls-base-colrow-ra100-wsfa200-116
Train Loss: 21.8736 | Aux Loss: 0.0316 | Base Loss: 2.8935 | Train Acc: 0.2883 | Valid Acc: 0.2784 | train_time: 2393.9s | val_time: 27.6s
Best Accuracy: 0.2784

### tianxianglii/cls-base-colrow-ra100-wsfa500-16
Train Loss: 26.9665 | Aux Loss: 0.0387 | Base Loss: 3.7362 | Train Acc: 0.1380 | Valid Acc: 0.1496 | train_time: 2401.1s | val_time: 27.7s
Best Accuracy: 0.1496
### tianxianglii/cls-base-colrow-ra100-wsfa500-116
Train Loss: 20.9839 | Aux Loss: 0.0302 | Base Loss: 2.8475 | Train Acc: 0.2956 | Valid Acc: 0.2886 | train_time: 2405.6s | val_time: 27.8s
Best Accuracy: 0.2984

### chencaihonga/cls-base-colrow-ra220-16
Train Loss: 27.5159 | Aux Loss: 0.0395 | Base Loss: 3.8425 | Train Acc: 0.1208 | Valid Acc: 0.1298 | train_time: 2426.7s | val_time: 27.6s
Best Accuracy: 0.1298
### chencaihonga/cls-base-colrow-ra220-116
rc_alpha=600.0, warmup_steps_for_aux=1
Epoch 34/130
Train Loss: 21.8485 | Aux Loss: 0.0316 | Base Loss: 2.8787 | Train Acc: 0.2894 | Valid Acc: 0.2818 | train_time: 2388.5s | val_time: 27.2s
Best Accuracy: 0.2858

## ra works
**ra600**: wsfa2000 > wsfa300 > wsfa700 > wsfa600 > wsfa3000 seems **totally random, not observable from just 17 epochs**
roseqw/cls-base-colrow-ra600-wsfa300-16
Train Loss: 26.6802 | Aux Loss: 0.0382 | Base Loss: 3.7646 | Train Acc: 0.1326 | Valid Acc: 0.1372 | train_time: 2382.2s | val_time: 27.1s
Best Accuracy: 0.1372

roseqw/cls-base-colrow-ra600-wsfa700-16
Train Loss: 27.6016 | Aux Loss: 0.0396 | Base Loss: 3.8434 | Train Acc: 0.1209 | Valid Acc: 0.1316 | train_time: 2373.4s | val_time: 26.9s
Best Accuracy: 0.1316

cdong121/cls-base-colrow-ra600-wsfa3000-16
Train Loss: 27.4458 | Aux Loss: 0.0392 | Base Loss: 3.9038 | Train Acc: 0.1151 | Valid Acc: 0.1228 | train_time: 2366.3s | val_time: 26.9s
Best Accuracy: 0.1228

b201xiaoli/cls-base-colrow-ra600-wsfa2000-16
Train Loss: 27.4417 | Aux Loss: 0.0394 | Base Loss: 3.8304 | Train Acc: 0.1224 | Valid Acc: 0.1398 | train_time: 2397.8s | val_time: 27.2s
Best Accuracy: 0.1398

**wsfa600**: ra100 > ra200 > ra300 > ra400 > ra500 > ra600 > ra700 > ra800 > ra900 > ra1000, **Early epochs might not decidable**
eastwangwei/cls-base-colrow-ra600-wsfa600-16
Train Loss: 27.3820 | Aux Loss: 0.0392 | Base Loss: 3.8332 | Train Acc: 0.1221 | Valid Acc: 0.1308 | train_time: 2413.4s | val_time: 27.7s
Best Accuracy: 0.1308

robinrainy/cls-base-colrow-ra800-wsfa600-16
Train Loss: 35.8763 | Aux Loss: 0.0398 | Base Loss: 4.0167 | Train Acc: 0.0980 | Valid Acc: 0.1126 | train_time: 2489.2s | val_time: 29.3s
Best Accuracy: 0.1126

robinrainy/cls-base-colrow-ra500-wsfa600-16
Train Loss: 23.1104 | Aux Loss: 0.0389 | Base Loss: 3.6675 | Train Acc: 0.1503 | Valid Acc: 0.1612 | train_time: 2376.3s | val_time: 27.0s
Best Accuracy: 0.1612

chencaihonga/cls-base-colrow-ra100-wsfa600-16
Train Loss: 6.4024 | Aux Loss: 0.0376 | Base Loss: 2.6449 | Train Acc: 0.3386 | Valid Acc: 0.3384 | train_time: 2370.4s | val_time: 27.1s
est Accuracy: 0.3384

djiangjiang/cls-base-colrow-ra200-wsfa600-16
Train Loss: 10.7605 | Aux Loss: 0.0385 | Base Loss: 3.0692 | Train Acc: 0.2499 | Valid Acc: 0.2578 | train_time: 2388.8s | val_time: 27.1s
Best Accuracy: 0.2578

hsdfuieqg/cls-base-colrow-ra300-wsfa600-16
Train Loss: 15.0341 | Aux Loss: 0.0390 | Base Loss: 3.3489 | Train Acc: 0.1992 | Valid Acc: 0.2082 | train_time: 2404.2s | val_time: 27.3s
Best Accuracy: 0.2082

hsdfuieqg/cls-base-colrow-ra400-wsfa600-16
Train Loss: 19.3042 | Aux Loss: 0.0394 | Base Loss: 3.5631 | Train Acc: 0.1602 | Valid Acc: 0.1698 | train_time: 2402.2s | val_time: 27.5s
Best Accuracy: 0.1698

asssmer/cls-base-colrow-ra700-wsfa600-16
Train Loss: 31.5430 | Aux Loss: 0.0395 | Base Loss: 3.8717 | Train Acc: 0.1176 | Valid Acc: 0.1234 | train_time: 2410.8s | val_time: 27.6s
Best Accuracy: 0.1234

autune/cls-base-colrow-ra900-wsfa600-16
Train Loss: 39.7775 | Aux Loss: 0.0398 | Base Loss: 3.9819 | Train Acc: 0.1038 | Valid Acc: 0.1104 | train_time: 2411.9s | val_time: 27.8s
Best Accuracy: 0.1104

b201xiaoli/cls-base-colrow-ra1000-wsfa600-16
Train Loss: 43.5181 | Aux Loss: 0.0395 | Base Loss: 4.0356 | Train Acc: 0.0991 | Valid Acc: 0.1088 | train_time: 2389.5s | val_time: 27.5s
Best Accuracy: 0.1088

**ra800**: wsfa800 > wsfa1000
gloden613/cls-base-colrow-ra800-wsfa800-16
Train Loss: 35.5920 | Aux Loss: 0.0395 | Base Loss: 3.9744 | Train Acc: 0.1063 | Valid Acc: 0.1150 | train_time: 2391.4s | val_time: 27.2s
Best Accuracy: 0.1150

gloden613/cls-base-colrow-ra800-wsfa1000-16
Train Loss: 35.8000 | Aux Loss: 0.0398 | Base Loss: 3.9896 | Train Acc: 0.1014 | Valid Acc: 0.1108 | train_time: 2402.4s | val_time: 27.4s
Best Accuracy: 0.1108




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