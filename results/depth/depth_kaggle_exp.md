lr=5e-5 < lr=1e-4

# lr ablation
## lr 1e-5 sh1weiwu/depth-base-none-lr1e-5-50
Valid AbsRel: 0.3198 | Valid L1: 1.4911 | Valid RMSE: 2.0734 | Valid a1: 0.6004
Valid AbsRel: 0.3211 | Valid L1: 1.5037 | Valid RMSE: 2.0840 | Valid a1: 0.5993
40802.9s	217	2026-01-24 02:26:21,153 - INFO -   Best a1:      0.6004 (Epoch 18)
40802.9s	218	2026-01-24 02:26:21,153 - INFO -   Best AbsRel:  0.3198 (Epoch 18)
40802.9s	219	2026-01-24 02:26:21,153 - INFO -   Best RMSE:    2.0734 (Epoch 18)

## sh1weiwu/depth-base-none-lr3e-5-50
Valid AbsRel: 0.3099 | Valid L1: 1.4539 | Valid RMSE: 2.0247 | Valid a1: 0.6201
40845.3s	217	2026-01-24 02:26:20,987 - INFO -   Best a1:      0.6204 (Epoch 15)
40845.3s	218	2026-01-24 02:26:20,988 - INFO -   Best AbsRel:  0.3076 (Epoch 16)
40845.3s	219	2026-01-24 02:26:20,988 - INFO -   Best RMSE:    2.0041 (Epoch 18)

## ly1122/depth-base-none-lr8e-5-50
Valid AbsRel: 0.3001 | Valid L1: 1.4219 | Valid RMSE: 1.9911 | Valid a1: 0.6263
40885.8s	217	2026-01-24 02:29:11,079 - INFO -   Best a1:      0.6312 (Epoch 12)
40885.8s	218	2026-01-24 02:29:11,079 - INFO -   Best AbsRel:  0.3001 (Epoch 19)
40885.8s	219	2026-01-24 02:29:11,079 - INFO -   Best RMSE:    1.9911 (Epoch 19)

## ly1122/depth-base-none-lr1e-4-50
Valid AbsRel: 0.3047 | Valid L1: 1.4272 | Valid RMSE: 1.9977 | Valid a1: 0.6249
40938.8s	217	2026-01-24 02:29:25,591 - INFO -   Best a1:      0.6249 (Epoch 19)
40938.8s	218	2026-01-24 02:29:25,591 - INFO -   Best AbsRel:  0.3038 (Epoch 15)
40938.8s	219	2026-01-24 02:29:25,591 - INFO -   Best RMSE:    1.9947 (Epoch 15)

## liucong126/depth-base-none-lr5e-5-50
Valid AbsRel: 0.3136 | Valid L1: 1.4808 | Valid RMSE: 2.0552 | Valid a1: 0.6142
41599.8s	216	2026-01-24 02:38:01,332 - INFO -   Best a1:      0.6142 (Epoch 19)
41599.8s	217	2026-01-24 02:38:01,332 - INFO -   Best AbsRel:  0.3136 (Epoch 19)
41599.8s	218	2026-01-24 02:38:01,332 - INFO -   Best RMSE:    2.0552 (Epoch 19)

## liucong126/depth-base-none-lr7e-5-50
alid AbsRel: 0.2983 | Valid L1: 1.4189 | Valid RMSE: 1.9872 | Valid a1: 0.6292
41489.0s	217	2026-01-24 02:35:14,116 - INFO -   Best a1:      0.6297 (Epoch 15)
41489.0s	218	2026-01-24 02:35:14,116 - INFO -   Best AbsRel:  0.2983 (Epoch 19)
41489.0s	219	2026-01-24 02:35:14,116 - INFO -   Best RMSE:    1.9772 (Epoch 18)


# rc weight ablation

### liucong12601/depth-base-colrow-rc30-50
Valid AbsRel: 0.2934 | Valid L1: 1.3688 | Valid RMSE: 1.9233 | Valid a1: 0.6378
41103.4s	217	2026-01-24 16:22:10,206 - INFO -   Best a1:      0.6378 (Epoch 19)
41103.4s	218	2026-01-24 16:22:10,206 - INFO -   Best AbsRel:  0.2934 (Epoch 19)
41103.4s	219	2026-01-24 16:22:10,206 - INFO -   Best RMSE:    1.9233 (Epoch 19)
### liucong12601/depth-base-colrow-rc70-50
Valid AbsRel: 0.2953 | Valid L1: 1.3796 | Valid RMSE: 1.9413 | Valid a1: 0.6367
41114.9s	217	2026-01-24 16:22:43,665 - INFO -   Best a1:      0.6397 (Epoch 18)
41114.9s	218	2026-01-24 16:22:43,665 - INFO -   Best AbsRel:  0.2929 (Epoch 18)
41114.9s	219	2026-01-24 16:22:43,666 - INFO -   Best RMSE:    1.9312 (Epoch 18)
### xulijuan/depth-base-colrow-rc50-50
Valid AbsRel: 0.2889 | Valid L1: 1.3636 | Valid RMSE: 1.9171 | Valid a1: 0.6439
41012.8s	217	2026-01-24 16:19:12,474 - INFO -   Best a1:      0.6439 (Epoch 19)
41012.8s	218	2026-01-24 16:19:12,474 - INFO -   Best AbsRel:  0.2889 (Epoch 19)
41012.8s	219	2026-01-24 16:19:12,474 - INFO -   Best RMSE:    1.9171 (Epoch 19)
### xulijuan/depth-base-colrow-rc150-50
Valid AbsRel: 0.2947 | Valid L1: 1.3478 | Valid RMSE: 1.8983 | Valid a1: 0.6425
41378.7s	217	2026-01-24 16:25:44,342 - INFO -   Best a1:      0.6425 (Epoch 19)
41378.7s	218	2026-01-24 16:25:44,342 - INFO -   Best AbsRel:  0.2938 (Epoch 16)
41378.7s	219	2026-01-24 16:25:44,342 - INFO -   Best RMSE:    1.8983 (Epoch 19)
### jacksisi/depth-base-colrow-rc120-50
Valid AbsRel: 0.2926 | Valid L1: 1.3479 | Valid RMSE: 1.8995 | Valid a1: 0.6432
40722.9s	217	2026-01-24 16:17:40,867 - INFO -   Best a1:      0.6432 (Epoch 19)
40722.9s	218	2026-01-24 16:17:40,867 - INFO -   Best AbsRel:  0.2926 (Epoch 19)
40722.9s	219	2026-01-24 16:17:40,868 - INFO -   Best RMSE:    1.8995 (Epoch 19)
### jacksisi/depth-base-colrow-rc200-50
Valid AbsRel: 0.2876 | Valid L1: 1.3322 | Valid RMSE: 1.8866 | Valid a1: 0.6466
40864.9s	217	2026-01-24 16:20:21,548 - INFO -   Best a1:      0.6466 (Epoch 19)
40864.9s	218	2026-01-24 16:20:21,548 - INFO -   Best AbsRel:  0.2876 (Epoch 19)
40864.9s	219	2026-01-24 16:20:21,548 - INFO -   Best RMSE:    1.8866 (Epoch 19)

# redo 20160126

## Epoch 80
### yangjiamin/depth-base-none-lr7e-5-650
train_loss	valid_abs_rel	valid_l1	valid_rmse	valid_a1	train_time	val_time	epoch
1.652472377	0.284699033	1.343795474	1.898940385	0.649295627	1982.193583	145.7565393	80

### zjl001/depth-base-none-d-451
1.690485835	0.28100252	1.344053163	1.903293421	0.655251558	1977.510468	139.2395663	80

### zjl001/depth-base-none-d-452
1.643542647	0.279665267	1.331945701	1.882138597	0.659114507	1976.351606	137.9383237	80

### smartchaochao/depth-base-none-d-453
1.645681024	0.280985886	1.338850402	1.896116697	0.652933387	1984.330497	145.863472	80

### zongjiaxin/depth-base-colrow-d-450
1.765725493	0.247899808	1.218496238	1.745054749	0.695210403	2002.491724	140.5714061	80

### denghaimeng/depth-base-colrow-d-451
1.761826634	0.255675134	1.235615613	1.777523699	0.686010109	1991.047552	152.0828071	80

### zzr123123/depth-base-colrow-d-452
1.837749958	0.259338521	1.27922171	1.816701944	0.679821441	1987.619475	145.1855648	80

### smartchaochao/depth-base-colrow-d-453
1.785965204	0.251169919	1.240191675	1.773891309	0.689955246	1991.920565	138.3228343	80

### denghaimeng/depth-base-abs-d-450
1.625218987	0.284939672	1.359443086	1.927397098	0.649387404	2005.339828	159.8240519	80

### jjjerry12138/depth-base-abs-d-451
1.645646334	0.283224298	1.345418401	1.901687535	0.654123684	1988.687811	147.9522591	80

### zhoujiahui0199/depth-base-abs-d-452
1.656977534	0.280499556	1.340444607	1.900754164	0.65813738	1976.696081	161.8105047	80

### cycyxcy/depth-base-abs-d-453
1.622940421	0.281587759	1.342394686	1.896823508	0.653632686	1981.950897	155.6544867	80

### zzr123123/depth-base-rope-d-450
1.693910599	0.300384011	1.449714727	2.023197931	0.625167206	2079.801165	144.9125423	80

### zongjiaxin/depth-base-rope-d-451
1.762308836	0.295646718	1.412757112	1.978943141	0.634286942	2072.600168	161.8100932	80

### yangjiamin/depth-base-rope-d-452
1.755584121	0.295037996	1.424593028	1.985672094	0.63457598	2080.58253	158.0702593	80

### yangjiamin/depth-base-rope-d-453
1.707805753	0.297335642	1.411016825	1.97488115	0.631965994	2077.542921	156.6169548	80

## Epoch 120

### yangyangchengcheng/depth-base-rope-d-753
1.577051878	0.298177865	1.431174888	1.997525026	0.631959746	2075.083171	151.7533829	120

### top1pcl/depth-base-rope-d-752
1.61135149	0.293467523	1.425510652	1.987345528	0.635995668	2071.721387	154.546138	120

### top1pcl/depth-base-rope-d-751
1.612473726	0.29487146	1.412834811	1.97930057	0.634397084	2070.546313	143.9494793	120

### srmmmm/depth-base-rope-d-750
1.551391602	0.297736152	1.431863945	2.000878175	0.630306182	2071.978714	147.8398447	120

### ttzyty/depth-base-colrow-d-750
1.567012429	0.245876605	1.212264093	1.73918571	0.698031086	2046.639598	196.9684005	120

### houwen/depth-base-colrow-d-751
1.551429033	0.253871708	1.24001927	1.777270397	0.689096949	1991.707285	154.9952583	120

### hysz0821/depth-base-colrow-d-752
1.615913272	0.254062326	1.246218872	1.780649683	0.688852239	1989.557887	156.4382522	120

### srmmmm/depth-base-colrow-d-753
1.574456215	0.247774132	1.222404548	1.757295309	0.694950032	1988.538512	168.8413153	120

### kanwenbin/depth-base-abs-d-750
1.493130565	0.282670542	1.354142432	1.920017225	0.652776343	1975.590351	144.0643332	120

### ttzyty/depth-base-abs-d-751
1.495172262	0.283674692	1.341291446	1.896320234	0.65564047	1978.223475	159.7986887	120

### houwen/depth-base-abs-d-752
1.513568759	0.281517242	1.339240333	1.898460285	0.657490853	1989.714127	162.7603753	120

### kanwenbin/depth-base-abs-d-753
1.492671609	0.282969948	1.352703065	1.911658169	0.651787837	1975.822342	164.6718507	120

### huoqiuxia/depth-base-none-lr7e-5-950
1.515484929	0.284260021	1.339188806	1.891333572	0.65114238	1984.765806	140.4672875	120

### huoqiuxia/depth-base-none-d-751
1.539426088	0.280753085	1.341114189	1.896318551	0.654537425	1981.191464	149.4550769	120

### yangyangchengcheng/depth-base-none-d-752
1.503038168	0.279387792	1.333261322	1.884978994	0.657784829	1992.680213	144.3659105	120

### hysz0821/depth-base-none-d-753
1.512086749	0.281743048	1.347906393	1.907516986	0.653853792	1990.299896	186.7125726	120

## lr ablation 
base lr 5e-5 seems best, with bs=24
namespace(train_roots=['/kaggle/input/hsm-train-part01', '/kaggle/input/hsm-train-part02', '/kaggle/input/hsm-train-part03', '/kaggle/input/hsm-train-part04', '/kaggle/input/hsm-train-part05'], eval_root='/kaggle/input/hsm-test-val', eval_split='test', model_type='dinov3', use_abs_pos_emb=False, use_rot_pos_emb=True, model_size='base', train_sizes=[(224, 224)], eval_size=(224, 224), final_eval_size=(224, 224), color_jitter_prob=0.5, scale_jitter=(1.0, 1.2), scale_jitter_sw=(1.0, 1.01), **batch_size=24, grad_accum_steps=4**, patch_size=16, **lr=0.00028**, lr_aux=1e-05, eta_min=1e-07, epochs=120, break_at_epoch=None, has_pos=False, weight_decay=0.05, overlap=0, seed=50, val_steps=None, use_rc_loss=False, loss_type='smooth_l1', rc_alpha=200.0, workers=2, composite_lr=True, warmup_steps=3000, warmup_ratio=None, clip_value=1.0, debug_loss_stats=False, debug_loss_interval=1, depth_decoder='dpt', log_interval=500, show_peak_gpu_mem=True, depth_eval_mode='relative', align_mode='mean_std', silog_w=0.0, depth_norm='median', ssim_norm_mode='per_image', ssim_percentiles=(5.0, 95.0), eval_crop_mode='crop', eval_dataset='hypersim', eval_depth_min=0.001, eval_depth_max=None, eval_prescale=1.07, train_depth_valid_thresh=0.1, eval_depth_valid_thresh=0.01, use_sliding_window=False, sw_window_size=None, sw_overlap=0.25, debug_dataset=False, output_dir='/kaggle/working', csv_interval=5, prefetch_factor=2, compile_model=False, save_full_ckpt=True, resume_full_ckpt=False, resume_ckpt_path=None, resume_args=True, resume_scheduler=True, resume_optimizer=False, resume_bs=True, resume_img_size=False, total_run_time_hr=12.0, train=True, val=False, final_use_sliding_window=True, final_sw_window_size=(224, 224), final_sw_overlap=0.25, cuda_alloc_conf='expandable_segments:True')
### xwj66666/depth-base-rope-lr0-00028-50
--- Epoch 19 Validation Summary ---
42560.0s	215	2026-01-30 01:04:04,130 - INFO -   Train Loss: 0.6234 | train_time: 2027.7s | val_time: 162.0s
Valid AbsRel: 0.2790 | Valid L1: 2.0210 | Valid RMSE: 2.7589 | Valid a1: 0.6492
42563.3s	224	2026-01-30 01:04:07,416 - INFO -   Best a1:      0.6511 (Epoch 17)
42563.3s	225	2026-01-30 01:04:07,416 - INFO -   Best AbsRel:  0.2784 (Epoch 17)
42563.3s	226	2026-01-30 01:04:07,417 - INFO -   Best RMSE:    2.6970 (Epoch 17)

### jiangshuai0210/depth-base-rope-lr2e-4-50
Valid AbsRel: 0.2764 | Valid L1: 1.9411 | Valid RMSE: 2.6692 | Valid a1: 0.6537
42078.7s	224	2026-01-30 01:18:45,285 - INFO -   Best a1:      0.6562 (Epoch 16)
42078.7s	225	2026-01-30 01:18:45,286 - INFO -   Best AbsRel:  0.2759 (Epoch 16)
42078.7s	226	2026-01-30 01:18:45,286 - INFO -   Best RMSE:    2.6692 (Epoch 19)

### keyongkk/depth-base-rope-lr30-50
Valid AbsRel: 0.2876 | Valid L1: 2.0291 | Valid RMSE: 2.7572 | Valid a1: 0.6422
42107.4s	224	2026-01-30 01:22:57,982 - INFO -   Best a1:      0.6535 (Epoch 17)
42107.4s	225	2026-01-30 01:22:57,982 - INFO -   Best AbsRel:  0.2801 (Epoch 17)
42107.4s	226	2026-01-30 01:22:57,982 - INFO -   Best RMSE:    2.7247 (Epoch 17)

### xwj66666/depth-base-rope-lr2-5e-4-50
Valid AbsRel: 0.2812 | Valid L1: 2.0210 | Valid RMSE: 2.7616 | Valid a1: 0.6442
42461.9s	224	2026-01-30 01:21:43,086 - INFO -   Best a1:      0.6503 (Epoch 17)
42461.9s	225	2026-01-30 01:21:43,086 - INFO -   Best AbsRel:  0.2775 (Epoch 15)
42461.9s	226	2026-01-30 01:21:43,086 - INFO -   Best RMSE:    2.6972 (Epoch 15)

### jiangshuai0210/depth-base-rope-lr18-50
Valid AbsRel: 0.2771 | Valid L1: 1.9952 | Valid RMSE: 2.7450 | Valid a1: 0.6512
42352.7s	214	2026-01-30 01:26:40,371 - INFO -   Best a1:      0.6530 (Epoch 17)
42352.7s	215	2026-01-30 01:26:40,371 - INFO -   Best AbsRel:  0.2768 (Epoch 17)
42352.7s	216	2026-01-30 01:26:40,371 - INFO -   Best RMSE:    2.6978 (Epoch 15)

### maoshuwen0415/depth-base-rope-lr32-50
Valid AbsRel: 0.2853 | Valid L1: 2.0325 | Valid RMSE: 2.7696 | Valid a1: 0.6438
41864.9s	224	2026-01-30 01:19:09,844 - INFO -   Best a1:      0.6506 (Epoch 18)
41864.9s	225	2026-01-30 01:19:09,844 - INFO -   Best AbsRel:  0.2800 (Epoch 18)
41864.9s	226	2026-01-30 01:19:09,845 - INFO -   Best RMSE:    2.7142 (Epoch 15)


# 20260201

- kernel_id: zhaotianchi/depth-base-none-gpu-616
0.504994094	0.263558595	1.915365367	2.648540554	0.668993971	1976.82016	185.7157304	130

- kernel_id: zhaotianchi/depth-base-none-gpu-617
0.503231406	0.265156475	1.936869075	2.667331517	0.666229313	1974.504129	157.4121177	130
(224, 224) 0.2651564746765398 0.6662293127970225

- kernel_id: smartchaochao/depth-base-none-gpu-618
train_loss	valid_abs_rel	valid_l1	valid_rmse	valid_a1	train_time	val_time	epoch
0.500846148	0.265182302	1.894364577	2.612784714	0.668594001	1966.862987	155.5967937	130

- kernel_id: ly1122/depth-base-none-gpu-619
0.505483329	0.270230147	1.960706166	2.693200568	0.662133434	1979.540688	178.5700157	130

- kernel_id: ly1122/depth-base-abs-gpu-616
0.502320588	0.266979984	1.947830614	2.68135967	0.665919515	2001.137455	176.2948356	130

- kernel_id: ycy1n66/depth-base-abs-gpu-617
0.501953959	0.269736714	1.981106429	2.720135506	0.660292323	1969.696164	169.8724654	130

- kernel_id: ssss7777/depth-base-abs-gpu-618
0.50170821	0.265769081	1.942398898	2.680596922	0.663796611	1969.988842	172.6861079	130

- kernel_id: ycy1n66/depth-base-abs-gpu-619
0.502655506	0.265277568	1.917212134	2.658940565	0.66536527	2002.451762	155.859839	130

- kernel_id: xuwenhui123/depth-base-rope-gpu-716
0.501743317	0.272217735	1.989055145	2.727916768	0.655464283	2077.735456	158.8814764	130

- kernel_id: zongjiaxin/depth-base-rope-gpu-717
0.50289005	0.270047718	1.989712181	2.718261821	0.657751917	2078.856191	163.99788	130

- kernel_id: sinayliu/depth-base-rope-gpu-718
0.502420962	0.273909399	2.015457488	2.753082419	0.656273619	2252.950933	157.4922948	130

- kernel_id: xuwenhui123/depth-base-rope-gpu-719
0.502644897	0.275077603	2.023072704	2.766485286	0.652405411	2085.343162	156.3873715	130


- kernel_id: xuwenhui123/depth-base-colrow-gpu-616
4.506750584	0.278552067	2.041110041	2.784216682	0.643435521	1980.740578	178.0014544	130

- kernel_id: xuwenhui123/depth-base-colrow-gpu-617
4.049808025	0.275616348	1.98281465	2.720840199	0.64987221	1979.349785	192.1549625	130



## ra ablation
**19 Epochs not informative**
cfy002/depth-base-colrow-ra200-wsfa600-16
Valid AbsRel: 0.2867 | Valid L1: 2.1052 | Valid RMSE: 2.8729 | Valid a1: 0.6345

cfy002/depth-base-colrow-ra100-wsfa600-16
Valid AbsRel: 0.2849 | Valid L1: 2.0504 | Valid RMSE: 2.7824 | Valid a1: 0.6422

cheaterchow/depth-base-colrow-ra150-wsfa600-16
Valid AbsRel: 0.2851 | Valid L1: 2.0515 | Valid RMSE: 2.7865 | Valid a1: 0.6404

cheaterchow/depth-base-colrow-ra250-wsfa600-16
Valid AbsRel: 0.2802 | Valid L1: 2.0579 | Valid RMSE: 2.8092 | Valid a1: 0.6447

chengchi1007/depth-base-colrow-ra50-wsfa600-16
Valid AbsRel: 0.2796 | Valid L1: 2.0368 | Valid RMSE: 2.7709 | Valid a1: 0.6446

chengchi1007/depth-base-colrow-ra80-wsfa600-16
Valid AbsRel: 0.2828 | Valid L1: 2.0447 | Valid RMSE: 2.7804 | Valid a1: 0.6439

chenhao1213/depth-base-colrow-ra30-wsfa600-16
Valid AbsRel: 0.2801 | Valid L1: 2.0365 | Valid RMSE: 2.7602 | Valid a1: 0.6444

chenhao1213/depth-base-colrow-ra400-wsfa600-16
Valid AbsRel: 0.2847 | Valid L1: 2.0458 | Valid RMSE: 2.7975 | Valid a1: 0.6393

cq1234/depth-base-colrow-ra500-wsfa600-16
Valid AbsRel: 0.2880 | Valid L1: 2.0614 | Valid RMSE: 2.8051 | Valid a1: 0.6387

cq1234/depth-base-colrow-ra600-wsfa600-16
Valid AbsRel: 0.2816 | Valid L1: 2.0482 | Valid RMSE: 2.7891 | Valid a1: 0.6426