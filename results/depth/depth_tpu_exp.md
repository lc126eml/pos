
### robinrainy/depth-base-rope-tpu8c-50
epoch 100
Valid AbsRel: 0.2737 | Valid L1: 1.9975 | Valid RMSE: 2.7361 | Valid a1: 0.6553
17342.5s	7299	2026-01-30 08:44:36,609 - INFO -   Best a1:      0.6567 (Epoch 95)
17342.5s	7300	2026-01-30 08:44:36,609 - INFO -   Best AbsRel:  0.2721 (Epoch 94)
17342.5s	7301	2026-01-30 08:44:36,609 - INFO -   Best RMSE:    2.7095 (Epoch 99)

###  robinrainy/depth-base-rope-tpu8c-51
Epoch 100 Validation Summary ---
16749.2s	791	2026-01-30 14:36:07,949 - INFO -   Train Loss: 0.3984 | train_time: 123.7s | val_time: 33.1s
16749.2s	792	2026-01-30 14:36:07,949 - INFO -  Valid AbsRel: 0.2692 | Valid L1: 1.9471 | Valid RMSE: 2.6847 | Valid a1: 0.6604
16749.2s	799	2026-01-30 14:36:08,024 - INFO -   Best a1:      0.6632 (Epoch 88)
16749.2s	800	2026-01-30 14:36:08,024 - INFO -   Best AbsRel:  0.2681 (Epoch 88)
16749.2s	801	2026-01-30 14:36:08,024 - INFO -   Best RMSE:    2.6531 (Epoch 94)

### xulijuan/depth-base-colrow-tpu8c-51
0.902304292	0.270149231	2.001711845	2.748797178	0.658001184	145.6667423	33.16564536	100
Valid AbsRel: 0.2701 | Valid L1: 2.0017 | Valid RMSE: 2.7488 | Valid a1: 0.6580
17224.2s	799	2026-01-30 16:24:25,134 - INFO -   Best a1:      0.6580 (Epoch 100)
17224.2s	800	2026-01-30 16:24:25,134 - INFO -   Best AbsRel:  0.2690 (Epoch 95)
17224.2s	801	2026-01-30 16:24:25,134 - INFO -   Best RMSE:    2.7219 (Epoch 66)

  - kernel_id: guyuefangyuan6666/depth-base-none-tpu8c-60
Valid AbsRel: 0.2719 | Valid L1: 1.9879 | Valid RMSE: 2.7296 | Valid a1: 0.6511
16520.8s	799	2026-02-04 08:08:12,057 - INFO -   Best a1:      0.6541 (Epoch 97)
16520.8s	800	2026-02-04 08:08:12,057 - INFO -   Best AbsRel:  0.2703 (Epoch 97)
16520.8s	801	2026-02-04 08:08:12,057 - INFO -   Best RMSE:    2.7135 (Epoch 97)
  - kernel_id: robinrainy/depth-base-abs-tpu8c-60
Valid AbsRel: 0.2724 | Valid L1: 2.0107 | Valid RMSE: 2.7515 | Valid a1: 0.6572
16379.4s	799	2026-02-04 07:10:45,498 - INFO -   Best a1:      0.6585 (Epoch 88)
16379.4s	800	2026-02-04 07:10:45,498 - INFO -   Best AbsRel:  0.2717 (Epoch 88)
16379.4s	801	2026-02-04 07:10:45,498 - INFO -   Best RMSE:    2.7261 (Epoch 88)
  - kernel_id: xulijuan/depth-base-rope-tpu8c-60
Valid AbsRel: 0.2753 | Valid L1: 2.0405 | Valid RMSE: 2.7940 | Valid a1: 0.6526
16771.6s	799	2026-02-04 07:17:17,585 - INFO -   Best a1:      0.6526 (Epoch 100)
16771.6s	800	2026-02-04 07:17:17,585 - INFO -   Best AbsRel:  0.2753 (Epoch 100)
16771.6s	801	2026-02-04 07:17:17,585 - INFO -   Best RMSE:    2.7689 (Epoch 43)
  - kernel_id: zhoushuqing/depth-base-colrow-tpu8c-60
Valid AbsRel: 0.2646 | Valid L1: 1.9668 | Valid RMSE: 2.6952 | Valid a1: 0.6614
16876.3s	799	2026-02-04 08:22:58,098 - INFO -   Best a1:      0.6640 (Epoch 97)
16876.3s	800	2026-02-04 08:22:58,098 - INFO -   Best AbsRel:  0.2642 (Epoch 97)
16876.3s	801	2026-02-04 08:22:58,098 - INFO -   Best RMSE:    2.6883 (Epoch 83)

# 20260205
namespace(train_roots=['/kaggle/input/hsm-train-part01', '/kaggle/input/hsm-train-part02', '/kaggle/input/hsm-train-part03', '/kaggle/input/hsm-train-part04', '/kaggle/input/hsm-train-part05'], eval_root='/kaggle/input/hsm-test-val', image_list_path='/kaggle/input/ds-file-list', eval_split='test', model_type='dinov3', use_abs_pos_emb=False, use_rot_pos_emb=False, model_size='base', train_sizes=[(224, 224)], eval_size=(224, 224), final_eval_size=(224, 224), color_jitter_prob=0.5, scale_jitter=(1.0, 1.2), scale_jitter_sw=(1.0, 1.01), batch_size=24, val_batch_size=16, val_drop_last=True, val_pad_to_full_batch=False, patch_size=16, **lr=0.0002, lr_aux=1e-05, eta_min=1e-07, epochs=120, break_at_epoch=120, has_pos=False, weight_decay=0.05**, overlap=0, seed=60, val_steps=None, use_rc_loss=True, loss_type='smooth_l1', **rc_alpha=100, warmup_steps_for_aux=600**, alpha_min=10, workers=0, tpu_workers=0, tpu_threads=1, composite_lr=True, warmup_steps=1500.0, warmup_ratio=None, clip_value=1.0, debug_loss_stats=False, debug_loss_interval=1, depth_decoder='dpt', log_interval=30000, show_peak_gpu_mem=False, log_all_ranks=False, debug_xla=False, debug_val_interval=50, debug_val_empty_limit=50, val_mark_step_interval=5, use_bf16=True, depth_eval_mode='relative', align_mode='mean_std', silog_w=0.0, grad_w=0.5, depth_norm='median', ssim_norm_mode='per_image', ssim_percentiles=(5.0, 95.0), eval_crop_mode='crop', eval_dataset='hypersim', eval_depth_min=0.001, eval_depth_max=None, eval_prescale=1.07, train_depth_valid_thresh=0.1, eval_depth_valid_thresh=0.01, min_valid_pixels=50, loss_det_threshold=1e-06, use_sliding_window=False, sw_window_size=None, sw_overlap=0.25, debug_dataset=False, output_dir='/kaggle/working', csv_interval=5, prefetch_factor=2, val_workers=None, val_prefetch_factor=1, val_persistent_workers=False, compile_model=False, save_full_ckpt=True, save_full_ckpt_interval=10, save_weights=False, resume_full_ckpt=False, resume_ckpt_path=None, resume_args=True, resume_scheduler=True, resume_optimizer=False, resume_bs=True, resume_img_size=False, total_run_time_hr=9.0, train=True, val=True, final_use_sliding_window=False, final_sw_window_size=(224, 224), final_sw_overlap=0.25)

  - kernel_id: qcrqcrqcr/depth-base-none-tpu8c-60
train_loss	valid_abs_rel	valid_l1	valid_rmse	valid_a1	train_time	val_time	epoch	base_loss	aux_loss
0.436068833	0.272498667	1.999241352	2.743818283	0.656368434	150.5239995	33.8353827	120
19475.0s	723	2026-02-04 19:41:38,371 - INFO -   Best a1:      0.6581 (Epoch 93)
19475.0s	724	2026-02-04 19:41:38,371 - INFO -   Best AbsRel:  0.2713 (Epoch 93)
19475.0s	725	2026-02-04 19:41:38,371 - INFO -   Best RMSE:    2.7151 (Epoch 65)
  - kernel_id: dingziheng11/depth-base-abs-tpu8c-60
0.436708897	0.270764738	1.981711984	2.725152016	0.660026968	172.7958477	32.94211388	120
19468.6s	723	2026-02-04 19:41:11,848 - INFO -   Best a1:      0.6623 (Epoch 92)
19468.6s	724	2026-02-04 19:41:11,848 - INFO -   Best AbsRel:  0.2688 (Epoch 80)
19468.6s	725	2026-02-04 19:41:11,848 - INFO -   Best RMSE:    2.6874 (Epoch 29)
  - kernel_id: zhoushuqing/depth-base-rope-tpu8c-60
0.428633898	0.275023699	2.000036478	2.741199017	0.651971042	172.2788739	35.39929914	120
19851.3s	723	2026-02-04 19:47:10,132 - INFO -   Best a1:      0.6544 (Epoch 97)
19851.3s	724	2026-02-04 19:47:10,132 - INFO -   Best AbsRel:  0.2731 (Epoch 87)
19851.3s	725	2026-02-04 19:47:10,132 - INFO -   Best RMSE:    2.7124 (Epoch 101)
  - kernel_id: guyuefangyuan6666/depth-base-colrow-tpu8c-60
1.089882731	0.28148362	2.022753	2.757255077	0.642692268	161.4566755	33.45825458	120	0.46462062	0.006252624
19606.5s	723	2026-02-04 19:42:47,949 - INFO -   Best a1:      0.6439 (Epoch 100)
19606.5s	724	2026-02-04 19:42:47,949 - INFO -   Best AbsRel:  0.2809 (Epoch 100)
19606.5s	725	2026-02-04 19:42:47,949 - INFO -   Best RMSE:    2.7475 (Epoch 77)

# redo 0205
namespace(train_roots=['/kaggle/input/hsm-train-part01', '/kaggle/input/hsm-train-part02', '/kaggle/input/hsm-train-part03', '/kaggle/input/hsm-train-part04', '/kaggle/input/hsm-train-part05'], eval_root='/kaggle/input/hsm-test-val', image_list_path='/kaggle/input/ds-file-list', eval_split='test', model_type='dinov3', use_abs_pos_emb=False, use_rot_pos_emb=False, model_size='base', train_sizes=[(224, 224)], eval_size=(224, 224), final_eval_size=(224, 224), color_jitter_prob=0.5, scale_jitter=(1.0, 1.2), scale_jitter_sw=(1.0, 1.01), batch_size=24, val_batch_size=16, val_drop_last=True, val_pad_to_full_batch=False, patch_size=16, **lr=0.0004, lr_aux=1e-05, eta_min=1e-06, epochs=100, break_at_epoch=None**, has_pos=False, weight_decay=0.05, overlap=0, seed=60, val_steps=None, use_rc_loss=True, loss_type='smooth_l1', rc_alpha=300, warmup_steps_for_aux=600, alpha_min=10, workers=0, tpu_workers=0, tpu_threads=1, composite_lr=True, warmup_steps=1500.0, warmup_ratio=None, clip_value=1.0, debug_loss_stats=False, debug_loss_interval=1, depth_decoder='dpt', log_interval=30000, show_peak_gpu_mem=False, log_all_ranks=False, debug_xla=False, debug_val_interval=50, debug_val_empty_limit=50, val_mark_step_interval=5, use_bf16=True, depth_eval_mode='relative', align_mode='mean_std', silog_w=0.0, grad_w=0.5, depth_norm='median', ssim_norm_mode='per_image', ssim_percentiles=(5.0, 95.0), eval_crop_mode='crop', eval_dataset='hypersim', eval_depth_min=0.001, eval_depth_max=None, eval_prescale=1.07, train_depth_valid_thresh=0.1, eval_depth_valid_thresh=0.01, min_valid_pixels=50, loss_det_threshold=1e-06, use_sliding_window=False, sw_window_size=None, sw_overlap=0.25, debug_dataset=False, output_dir='/kaggle/working', csv_interval=5, prefetch_factor=2, val_workers=None, val_prefetch_factor=1, val_persistent_workers=False, compile_model=False, save_full_ckpt=True, save_full_ckpt_interval=10, save_weights=False, resume_full_ckpt=False, resume_ckpt_path=None, resume_args=True, resume_scheduler=True, resume_optimizer=False, resume_bs=True, resume_img_size=False, total_run_time_hr=9.0, train=True, val=False, final_use_sliding_window=False, final_sw_window_size=(224, 224), final_sw_overlap=0.25)

###
- kernel_id: dingziheng11/depth-base-colrow-ra300-lr20-60
train_loss	valid_abs_rel	valid_l1	valid_rmse	valid_a1	train_time	val_time	epoch	base_loss	aux_loss
1.037606001	0.277864546	2.016599417	2.763135195	0.648624778	206.5120049	33.943923	100	0.429776311	0.002026099
16628.3s	618	2026-02-05 06:02:09,333 - INFO -   Best a1:      0.6498 (Epoch 87)
16628.3s	619	2026-02-05 06:02:09,333 - INFO -   Best AbsRel:  0.2775 (Epoch 93)
16628.3s	620	2026-02-05 06:02:09,334 - INFO -   Best RMSE:    2.7563 (Epoch 83)
- kernel_id: qcrqcrqcr/depth-base-rope-ra300-lr20-60
0.427699417	0.275752902	2.027124643	2.781318665	0.651038647	177.9818141	34.66921544	100
16701.8s	618	2026-02-05 06:08:57,409 - INFO -   Best a1:      0.6524 (Epoch 79)
16701.8s	619	2026-02-05 06:08:57,409 - INFO -   Best AbsRel:  0.2750 (Epoch 79)
16701.8s	620	2026-02-05 06:08:57,409 - INFO -   Best RMSE:    2.7540 (Epoch 29)
- kernel_id: liucong126/depth-base-colrow-ra500-lr20-50
1.406970978	0.27871272	2.037115812	2.782590866	0.644919813	198.102236	34.93992376	100
16573.6s	618	2026-02-05 06:08:30,780 - INFO -   Best a1:      0.6459 (Epoch 78)
16573.6s	619	2026-02-05 06:08:30,780 - INFO -   Best AbsRel:  0.2779 (Epoch 91)
16573.6s	620	2026-02-05 06:08:30,780 - INFO -   Best RMSE:    2.7745 (Epoch 65)

  - kernel_id: zhoushuqing/depth-base-colrow-ra300-lr10-60
3.923696041	0.286315322	2.06047225	2.802691221	0.636708736	208.1507568	33.59880567	100
18022.8s	618	2026-02-05 06:24:41,267 - INFO -   Best a1:      0.6397 (Epoch 69)
18022.8s	619	2026-02-05 06:24:41,267 - INFO -   Best AbsRel:  0.2835 (Epoch 69)
18022.8s	620	2026-02-05 06:24:41,267 - INFO -   Best RMSE:    2.7754 (Epoch 69)

  - kernel_id: dingziheng11/depth-base-colrow-ra50-lr20-50
0.497936457	0.267750889	2.007124901	2.761988878	0.658494949	219.5798666	34.00298977	100
20331.1s	618	2026-02-05 12:25:00,264 - INFO -   Best a1:      0.6591 (Epoch 88)
20331.1s	619	2026-02-05 12:25:00,264 - INFO -   Best AbsRel:  0.2672 (Epoch 92)
20331.1s	620	2026-02-05 12:25:00,264 - INFO -   Best RMSE:    2.7404 (Epoch 38)
  - kernel_id: qcrqcrqcr/depth-base-colrow-ra100-lr20-50
16773.7s	609	2026-02-05 11:26:19,466 - INFO -   Train Loss: 0.5992 | aux_loss: 0.0018 | base_loss: 0.4226 | train_time: 189.1s | val_time: 33.4s
16773.7s	610	2026-02-05 11:26:19,466 - INFO -  Valid AbsRel: 0.2691 | Valid L1: 1.9310 | Valid RMSE: 2.6769 | Valid a1: 0.6590
16774.5s	618	2026-02-05 11:26:20,217 - INFO -   Best a1:      0.6594 (Epoch 91)
16774.5s	619	2026-02-05 11:26:20,217 - INFO -   Best AbsRel:  0.2689 (Epoch 89)
16774.5s	620	2026-02-05 11:26:20,217 - INFO -   Best RMSE:    2.6646 (Epoch 77)

  - kernel_id: liucong126/depth-base-colrow-ra70-lr30-50
0.463490725	0.265130758	1.994367957	2.739537716	0.665499508	244.7221169	36.80281162	100	0.398912609	0.000922542
19887.4s	598	2026-02-05 18:09:48,403 - INFO -   Best a1:      0.6670 (Epoch 72)
19887.4s	599	2026-02-05 18:09:48,403 - INFO -   Best AbsRel:  0.2635 (Epoch 72)
19887.4s	600	2026-02-05 18:09:48,403 - INFO -   Best RMSE:    2.7133 (Epoch 69)
  - kernel_id: qcrqcrqcr/depth-base-colrow-ra30-lr20-50
0.459581494	0.261092603	1.946790695	2.6661129	0.666600347	235.5085227	36.96928644	100	0.402469724	0.001903726
20587.2s	598	2026-02-05 17:27:17,272 - INFO -   Best a1:      0.6678 (Epoch 91)
20587.2s	599	2026-02-05 17:27:17,272 - INFO -   Best AbsRel:  0.2603 (Epoch 88)
20587.2s	600	2026-02-05 17:27:17,272 - INFO -   Best RMSE:    2.6527 (Epoch 70)

  - kernel_id: liucong126/depth-base-colrow-ra30-50
lr30
0.439899176	0.264034212	1.980457902	2.722648859	0.665266097	265.2922492	33.71172833	100	0.395024478	0.001495817
30520.3s	598	2026-02-06 09:44:23,230 - INFO -   Best a1:      0.6661 (Epoch 94)
30520.3s	599	2026-02-06 09:44:23,230 - INFO -   Best AbsRel:  0.2630 (Epoch 77)
30520.3s	600	2026-02-06 09:44:23,230 - INFO -   Best RMSE:    2.7120 (Epoch 86)

## 0206
Arguments: namespace(train_roots=['/kaggle/input/hsm-train-part01', '/kaggle/input/hsm-train-part02', '/kaggle/input/hsm-train-part03', '/kaggle/input/hsm-train-part04', '/kaggle/input/hsm-train-part05'], eval_root='/kaggle/input/hsm-test-val', image_list_path='/kaggle/input/ds-file-list', eval_split='test', model_type='dinov3', use_abs_pos_emb=False, use_rot_pos_emb=False, model_size='base', train_sizes=[(224, 224)], eval_size=(224, 224), final_eval_size=(224, 224), color_jitter_prob=0.5, scale_jitter=(1.0, 1.2), scale_jitter_sw=(1.0, 1.01), batch_size=16, val_batch_size=16, val_drop_last=True, val_pad_to_full_batch=False, patch_size=16, lr=0.00028, lr_aux=1e-05, eta_min=1e-07, epochs=100, break_at_epoch=None, has_pos=False, weight_decay=0.05, overlap=0, seed=50, val_steps=None, use_rc_loss=True, loss_type='smooth_l1', rc_alpha=200, warmup_steps_for_aux=600, alpha_min=10, workers=0, tpu_workers=0, tpu_threads=1, composite_lr=True, warmup_steps=562.5, warmup_ratio=None, clip_value=1.0, debug_loss_stats=False, debug_loss_interval=1, depth_decoder='dpt', log_interval=30000, show_peak_gpu_mem=False, log_all_ranks=False, debug_xla=False, debug_val_interval=50, debug_val_empty_limit=50, val_mark_step_interval=5, use_bf16=True, depth_eval_mode='relative', align_mode='mean_std', silog_w=0.0, grad_w=0.5, depth_norm='median', ssim_norm_mode='per_image', ssim_percentiles=(5.0, 95.0), eval_crop_mode='crop', eval_dataset='hypersim', eval_depth_min=0.001, eval_depth_max=None, eval_prescale=1.07, train_depth_valid_thresh=0.1, eval_depth_valid_thresh=0.01, min_valid_pixels=50, loss_det_threshold=1e-06, use_sliding_window=False, sw_window_size=None, sw_overlap=0.25, debug_dataset=False, output_dir='/kaggle/working', csv_interval=5, prefetch_factor=2, val_workers=None, val_prefetch_factor=1, val_persistent_workers=False, compile_model=False, save_full_ckpt=True, save_full_ckpt_interval=10, save_weights=False, resume_full_ckpt=False, resume_ckpt_path=None, resume_args=True, resume_scheduler=True, resume_optimizer=False, resume_bs=True, resume_img_size=False, total_run_time_hr=9.0, train=True, val=False, final_use_sliding_window=False, final_sw_window_size=(224, 224), final_sw_overlap=0.25)
  - kernel_id: xulijuan/depth-base-colrow-tpu-50
0.799292564	0.275224537	1.995040417	2.729932785	0.648314476	150.3905327	24.22531891	100	0.431728214	0.00183782
18391.2s	582	2026-02-06 15:15:24,725 - INFO -   Best a1:      0.6499 (Epoch 77)
18391.2s	583	2026-02-06 15:15:24,725 - INFO -   Best AbsRel:  0.2743 (Epoch 77)
18391.2s	584	2026-02-06 15:15:24,725 - INFO -   Best RMSE:    2.7076 (Epoch 56)

  - kernel_id: robinrainy/depth-base-rope-tpu-50
0.431771934	0.276929379	2.006662846	2.745367527	0.649701774	154.8535388	25.36235785	100
18947.1s	582	2026-02-06 15:24:53,825 - INFO -   Best a1:      0.6516 (Epoch 83)
18947.1s	583	2026-02-06 15:24:53,825 - INFO -   Best AbsRel:  0.2763 (Epoch 83)
18947.1s	584	2026-02-06 15:24:53,825 - INFO -   Best RMSE:    2.7239 (Epoch 83)
  - kernel_id: zhoushuqing/depth-base-abs-tpu-50
0.432593614	0.277951956	2.035669327	2.790364265	0.650841355	146.2642903	24.06896591	100
18984.2s	582	2026-02-06 15:25:49,438 - INFO -   Best a1:      0.6526 (Epoch 80)
18984.2s	583	2026-02-06 15:25:49,438 - INFO -   Best AbsRel:  0.2767 (Epoch 80)
18984.2s	584	2026-02-06 15:25:49,438 - INFO -   Best RMSE:    2.7669 (Epoch 59)
  - kernel_id: dingziheng11/depth-base-none-tpu-50
0.435297668	0.275605083	2.030567646	2.773960114	0.649367571	145.2373526	24.3258996	100
17976.8s	582	2026-02-06 15:09:17,933 - INFO -   Best a1:      0.6502 (Epoch 80)
17976.8s	583	2026-02-06 15:09:17,933 - INFO -   Best AbsRel:  0.2749 (Epoch 84)
17976.8s	584	2026-02-06 15:09:17,933 - INFO -   Best RMSE:    2.7460 (Epoch 44)
  - kernel_id: liucong126/depth-base-colrow-ra60-wsfa60-50
0.531631529	0.270258307	1.976840258	2.702106476	0.656248391	152.2008018	24.24135447	100	0.427662045	0.001732823
18471.2s	582	2026-02-06 15:19:19,929 - INFO -   Best a1:      0.6571 (Epoch 95)
18471.2s	583	2026-02-06 15:19:19,929 - INFO -   Best AbsRel:  0.2695 (Epoch 92)
18471.2s	584	2026-02-06 15:19:19,929 - INFO -   Best RMSE:    2.6820 (Epoch 74)

  - kernel_id: guyuefangyuan6666/depth-base-colrow-tpu-16
  batch_size=24, val_batch_size=16, val_drop_last=False, val_pad_to_full_batch=True, patch_size=16, lr=0.0002, lr_aux=4e-05, eta_min=1e-07, epochs=100, break_at_epoch=None, has_pos=False, weight_decay=0.05, overlap=0, seed=16
14895.2s	777	--- Epoch 95 Validation Summary ---
14895.2s	778	2026-02-07 08:46:40,106 - INFO -   Train Loss: 1.8059 | aux_loss: 0.0065 | base_loss: 0.4980 | train_time: 122.7s | val_time: 24.6s
14895.2s	779	2026-02-07 08:46:40,106 - INFO -  Valid AbsRel: 0.2891 | Valid L1: 2.0613 | Valid RMSE: 2.8070 | Valid a1: 0.6385
  - kernel_id: xulijuan/depth-base-colrow-ra70-wsfa60-lr7-16
- base_rc_True_lr27_relative_median_dec_dpt_h224w224_s16_alpha_70.csv
  batch_size=16, val_batch_size=16, val_drop_last=False, val_pad_to_full_batch=True, patch_size=16, **lr=0.00028**, lr_aux=4e-05, eta_min=1e-07, **epochs=130, break_at_epoch=100**, has_pos=False, weight_decay=0.05, overlap=0, seed=16, val_steps=None, use_rc_loss=True, loss_type='smooth_l1', rc_alpha=70, warmup_steps_for_aux=60, alpha_min=10, workers=0, tpu_workers=0, tpu_threads=1, composite_lr=True, warmup_steps=562.5,
20653.5s	583	2026-02-07 18:36:50,003 - INFO -   Best a1:      0.6653 (Epoch 100)
20653.5s	584	2026-02-07 18:36:50,003 - INFO -   Best AbsRel:  0.2644 (Epoch 100)
20653.5s	585	2026-02-07 18:36:50,003 - INFO -   Best RMSE:    2.6711 (Epoch 68)
  - kernel_id: guyuefangyuan6666/depth-base-colrow-ra30-wsfa60-lr7-16
  - base_rc_True_lr27_relative_median_dec_dpt_h224w224_s16_alpha_30.csv
  batch_size=16, val_batch_size=16, val_drop_last=False, val_pad_to_full_batch=True, patch_size=16, lr=0.00028, lr_aux=4e-05, eta_min=1e-07, epochs=130, break_at_epoch=100, has_pos=False, weight_decay=0.05, overlap=0, seed=16, val_steps=None, use_rc_loss=True, loss_type='smooth_l1', **rc_alpha=30**, warmup_steps_for_aux=60, alpha_min=10, workers=0, tpu_workers=0, tpu_threads=1, composite_lr=True, warmup_steps=562.5
18953.9s	583	2026-02-07 18:06:18,663 - INFO -   Best a1:      0.6720 (Epoch 86)
18953.9s	584	2026-02-07 18:06:18,663 - INFO -   Best AbsRel:  0.2575 (Epoch 86)
18953.9s	585	2026-02-07 18:06:18,663 - INFO -   Best RMSE:    2.6575 (Epoch 86)
# lr ablation

## lr: 1.5e-4 weight_decay=0.01
4868.0s	282	--- Epoch 27 Validation Summary ---
4868.0s	283	2026-02-04 10:13:52,915 - INFO -   Train Loss: 0.6400 | train_time: 123.6s | val_time: 31.9s
4868.0s	284	2026-02-04 10:13:52,915 - INFO -  Valid AbsRel: 0.2901 | Valid L1: 2.0129 | Valid RMSE: 2.7630 | Valid a1: 0.6365

## lr: 1.5e-4 weight_decay=0.05
4888.0s	282	--- Epoch 27 Validation Summary ---
4888.0s	283	2026-02-04 10:16:05,437 - INFO -   Train Loss: 0.6343 | train_time: 123.6s | val_time: 32.0s
4888.0s	284	2026-02-04 10:16:05,437 - INFO -  Valid AbsRel: 0.2874 | Valid L1: 2.0421 | Valid RMSE: 2.7787 | Valid a1: 0.6340

## lr: 1.0e-4 weight_decay=0.01
4912.1s	568	--- Epoch 27 Validation Summary ---
4912.1s	569	2026-02-04 10:12:33,889 - INFO -   Train Loss: 0.6581 | train_time: 122.9s | val_time: 31.9s
4912.1s	570	2026-02-04 10:12:33,889 - INFO -  Valid AbsRel: 0.2869 | Valid L1: 1.9755 | Valid RMSE: 2.7141 | Valid a1: 0.6398

## lr: 1e-4 weight_decay=0.05
4837.2s	229	--- Epoch 27 Validation Summary ---
4837.2s	230	2026-02-04 11:52:20,629 - INFO -   Train Loss: 0.6558 | train_time: 121.7s | val_time: 31.9s
4837.2s	231	2026-02-04 11:52:20,629 - INFO -  Valid AbsRel: 0.2837 | Valid L1: 1.9643 | Valid RMSE: 2.7007 | Valid a1: 0.6452

5301.0s	244	--- Epoch 30 Validation Summary ---
5301.0s	245	2026-02-04 12:00:04,424 - INFO -   Train Loss: 0.6346 | train_time: 121.4s | val_time: 31.8s
5301.0s	246	2026-02-04 12:00:04,424 - INFO -  Valid AbsRel: 0.2845 | Valid L1: 2.0224 | Valid RMSE: 2.7720 | Valid a1: 0.6452

## lr: 8e-5 weight_decay=0.05
4829.3s	229	--- Epoch 27 Validation Summary ---
4829.3s	230	2026-02-04 11:52:18,255 - INFO -   Train Loss: 0.6746 | train_time: 120.6s | val_time: 31.9s
4829.3s	231	2026-02-04 11:52:18,255 - INFO -  Valid AbsRel: 0.2874 | Valid L1: 2.0019 | Valid RMSE: 2.7414 | Valid a1: 0.6392

5288.4s	244	--- Epoch 30 Validation Summary ---
5288.4s	245	2026-02-04 11:59:57,325 - INFO -   Train Loss: 0.6530 | train_time: 120.0s | val_time: 31.8s
5288.4s	246	2026-02-04 11:59:57,325 - INFO -  Valid AbsRel: 0.2853 | Valid L1: 2.0113 | Valid RMSE: 2.7647 | Valid a1: 0.6456

## lr: 5e-5 weight_decay=0.05
4843.6s	229	--- Epoch 27 Validation Summary ---
4843.6s	230	2026-02-04 11:52:32,951 - INFO -   Train Loss: 0.7248 | train_time: 122.7s | val_time: 31.9s
4843.6s	231	2026-02-04 11:52:32,951 - INFO -  Valid AbsRel: 0.2932 | Valid L1: 2.0148 | Valid RMSE: 2.7565 | Valid a1: 0.6337

5306.0s	244	--- Epoch 30 Validation Summary ---
5306.0s	245	2026-02-04 12:00:15,267 - INFO -   Train Loss: 0.7078 | train_time: 121.2s | val_time: 31.7s
5306.0s	246	2026-02-04 12:00:15,267 - INFO -  Valid AbsRel: 0.2936 | Valid L1: 2.0855 | Valid RMSE: 2.8299 | Valid a1: 0.6337
5306.3s	255	2026-02-04 12:00:15,615 - INFO -   Best a1:      0.6363 (Epoch 26)
5306.3s	256	2026-02-04 12:00:15,616 - INFO -   Best AbsRel:  0.2895 (Epoch 26)
5306.3s	257	2026-02-04 12:00:15,616 - INFO -   Best RMSE:    2.7543 (Epoch 26)


## lr: 1e-4 weight_decay=0.05 rc_alpha=100
5366.1s	244	--- Epoch 30 Validation Summary ---
5366.1s	245	2026-02-04 14:02:05,953 - INFO -   Train Loss: 3.6477 | aux_loss: 0.0299 | base_loss: 0.6607 | train_time: 125.5s | val_time: 31.8s
5366.1s	246	2026-02-04 14:02:05,953 - INFO -  Valid AbsRel: 0.2945 | Valid L1: 2.0952 | Valid RMSE: 2.8565 | Valid a1: 0.6273
## lr: 1e-4 weight_decay=0.05 rc_alpha=200
5433.1s	244	--- Epoch 30 Validation Summary ---
5433.1s	245	2026-02-04 14:03:20,114 - INFO -   Train Loss: 6.6998 | aux_loss: 0.0302 | base_loss: 0.6612 | train_time: 128.4s | val_time: 32.2s
5433.1s	246	2026-02-04 14:03:20,114 - INFO -  Valid AbsRel: 0.3012 | Valid L1: 2.1038 | Valid RMSE: 2.8522 | Valid a1: 0.6199
## lr: 1e-4 weight_decay=0.05 rc_alpha=300
5386.5s	244	--- Epoch 30 Validation Summary ---
5386.5s	245	2026-02-04 14:02:45,518 - INFO -   Train Loss: 9.7683 | aux_loss: 0.0303 | base_loss: 0.6642 | train_time: 126.5s | val_time: 32.0s
5386.5s	246	2026-02-04 14:02:45,518 - INFO -  Valid AbsRel: 0.3012 | Valid L1: 2.1057 | Valid RMSE: 2.8570 | Valid a1: 0.6209

## lr: 1e-4 weight_decay=0.05 rc_alpha=400
5363.3s	244	--- Epoch 30 Validation Summary ---
5363.3s	245	2026-02-04 14:02:38,891 - INFO -   Train Loss: 12.8270 | aux_loss: 0.0304 | base_loss: 0.6673 | train_time: 124.4s | val_time: 31.8s
5363.3s	246	2026-02-04 14:02:38,891 - INFO -  Valid AbsRel: 0.2974 | Valid L1: 2.0869 | Valid RMSE: 2.8313 | Valid a1: 0.6234


## lr: 1e-4 weight_decay=0.05 rc_alpha=500
5373.2s	244	--- Epoch 30 Validation Summary ---
5373.2s	245	2026-02-04 14:03:01,903 - INFO -   Train Loss: 15.8835 | aux_loss: 0.0304 | base_loss: 0.6659 | train_time: 127.0s | val_time: 31.9s
5373.2s	246	2026-02-04 14:03:01,903 - INFO -  Valid AbsRel: 0.3012 | Valid L1: 2.1218 | Valid RMSE: 2.8765 | Valid a1: 0.6205

