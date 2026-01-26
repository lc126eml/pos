
# lr=0.0005 scale_jitter=(1.0, None), use_cat_max_ratio=False, ms_scales=(0.75, 1.0, 1.25)
## zongjiaxin/seg-base-colrow228
0.653585494	0.185792163	1174.78	27.36286902
final_ms_flip_acc	final_ms_flip_miou	best_ms_flip_acc	best_ms_flip_miou
0.663645387	0.188967779	0.663694918	0.189292639

## wzywzy1227/seg-base-colrow229
4.311634541	0.82439518
0.657858908	0.184681788	1281.766028	29.65322256
final_ms_flip_acc	final_ms_flip_miou
0.667701364	0.187183306
namespace(model_type='dinov3', use_abs_pos_emb=False, use_rot_pos_emb=False, model_size='base', num_classes=150, batch_size=16, grad_accum_steps=2, train_img_size=336, eval_img_size=336, use_ms_flip_eval=False, scale_jitter=(1.0, None), use_cat_max_ratio=False, cat_max_ratio=0.75, cat_max_ratio_tries=10, ms_scales=(0.75, 1.0, 1.25), eval_crop_mode='crop_or_pad', final_ms_flip_eval=True, lr=0.0005, lr_aux=1e-05, eta_min=1e-08, composite_lr=True, warmup_steps=3000, weight_decay=0.01, epochs=130, overlap=0, start_epoch=0, seed=29, use_rc_loss=True, huber_beta=0.1, rc_alpha=70.0, seg_head='upernet', feature_layers=[2, 5, 8, 11], workers=2, color_jitter={'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.05}, color_jitter_prob=0.1, train=True, val=False, ckpt_path=None, lock=False, clip_value=1.0, output_dir='/kaggle/working', log_interval=1, csv_interval=3, show_peak_gpu_mem=True, compile_model=False, save_full_ckpt=True, resume_full_ckpt=True, resume_ckpt_path='/kaggle/input/seg-base-colrow329/ckpt/last.pth', resume_scheduler=True, resume_optimizer=True, resume_bs=True, total_run_time_hr=11.1, base_path='/kaggle/input/ade20k-dataset/ADEChallengeData2016', pos_type=None)
## zongjiaxin/seg-base-abs228
0.627872765	0.155408517	1179.474867	28.18994498
final_ms_flip_acc	final_ms_flip_miou
0.640323222	0.159787223

## zhikaiwang/seg-base-abs229
0.686795771	0.79947722
0.631499946	0.161438748	1268.152841	28.99576473
final_ms_flip_acc	final_ms_flip_miou
0.641221285	0.163097382

## denghaimeng/seg-base-none28
0.62384367	0.151998833	1174.45207	27.59525752
final_ms_flip_acc	final_ms_flip_miou	best_ms_flip_acc	best_ms_flip_miou
0.638668299	0.156379253	0.638990939	0.157218933

## wzywzy1227/seg-base-none229
0.71562618	0.791805744
0.628263116	0.158210084	1431.396715	38.52299523
final_ms_flip_acc	final_ms_flip_miou
0.640283704	0.15904057


## denghaimeng/seg-base-rope228
Train Loss: 0.8367 | Train Acc: 0.7601 | Valid Acc: 0.6158 | Valid mIoU: 0.1442 | train_time: 1245.8s | val_time: 29.7s
final_ms_flip_acc	final_ms_flip_miou
0.625871599	0.146116465

## zhikaiwang/seg-base-rope229
0.788868546	0.772686005	
0.618389904	0.149044037	1343.998852	31.75034142
final_ms_flip_acc	final_ms_flip_miou
0.628063202	0.151884228

# lr 1e-05 bad
## autune/seg-base-colrow150
15.4296627	0.629720211	0.58590287	0.089508615	1271.957267	35.71642709
final_ms_flip_acc	final_ms_flip_miou
0.594150126	0.090602025
namespace(model_type='dinov3', use_abs_pos_emb=False, use_rot_pos_emb=False, model_size='base', num_classes=150, batch_size=16, grad_accum_steps=2, train_img_size=336, eval_img_size=368, use_ms_flip_eval=False, scale_jitter=(1.0, None), use_cat_max_ratio=False, cat_max_ratio=0.75, cat_max_ratio_tries=10, ms_scales=(0.75, 1.0, 1.25), eval_crop_mode='crop_or_pad', final_ms_flip_eval=True, lr=1e-05, lr_aux=1e-05, eta_min=5e-06, composite_lr=True, warmup_steps=500, weight_decay=0.01, epochs=130, overlap=0, start_epoch=0, seed=50, use_rc_loss=True, use_patch_position_loss=False, huber_beta=0.1, rc_alpha=70.0, seg_head='upernet', feature_layers=[2, 5, 8, 11], workers=2, color_jitter={'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.05}, color_jitter_prob=0.1, train=True, val=False, ckpt_path=None, lock=False, clip_value=1.0, output_dir='/kaggle/working', log_interval=1, csv_interval=3, show_peak_gpu_mem=True, compile_model=False, save_full_ckpt=True, resume_full_ckpt=True, resume_ckpt_path='/kaggle/input/seg-base-colrow50/ckpt/last.pth', resume_scheduler=True, resume_optimizer=True, resume_bs=True, total_run_time_hr=12.0, base_path='/kaggle/input/ade20k-dataset/ADEChallengeData2016', pos_type=None)
## autune/seg-base-rope150
1.335483313	0.651289463	0.589639306	0.095459089	1340.986082	39.44862175
final_ms_flip_acc	final_ms_flip_miou
0.597510278	0.096028201

# scale_jitter=(1.0, 1.3)
## denghaimeng/seg-base-rope126 eta_min=5e-06
1.560489296913147 0.5999572277069092 0.5791658163070679 0.08207729458808899
1351.0027208328247
40.206724643707275
57
71991
# scale_jitter=(1.0, 1.3) ms_scales=(0.9, 1.0, 1.15) better
## lr 1e-5 eta_min=1e-07 
zjl001/seg-base-rope120
Epoch 57/130 Summary
Train Loss: 1.4107 | Train Acc: 0.6278 | Valid Acc: 0.5952 | Valid mIoU: 0.0986 | train_time: 1353.6s | val_time: 39.8s 57

## lr 5e-6
zjl001/seg-base-rope21
Epoch 57/130 Summary
Train Loss: 1.6729 | Train Acc: 0.5777 | Valid Acc: 0.5653 | Valid mIoU: 0.0693 | train_time: 1350.6s | val_time: 39.4s 57


## use_cat_max_ratio=True

### autune/seg-base-none-lr1e-4-60
Train Loss: 1.3852 | Train Acc: 0.6223 | Valid Acc: 0.6040 | Valid mIoU: 0.1208 | train_time: 1272.1s | val_time: 36.4s
Best MS+Flip Acc: 0.6173 | Best MS+Flip mIoU: 0.1248
### autune/seg-base-none-lr2e-4-60
Train Loss: 1.4330 | Train Acc: 0.6110 | Valid Acc: 0.6009 | Valid mIoU: 0.1190 | train_time: 1270.4s | val_time: 36.3s
Best MS+Flip Acc: 0.6056 | Best MS+Flip mIoU: 0.1229
### jcy666/seg-base-none-lr3e-4-60
Train Loss: 1.4577 | Train Acc: 0.6054 | Valid Acc: 0.5975 | Valid mIoU: 0.1170 | train_time: 1278.6s | val_time: 36.9s
Best MS+Flip Acc: 0.6033 | Best MS+Flip mIoU: 0.1196
### jcy666/seg-base-none-lr3e-5-60
Train Loss: 1.4217 | Train Acc: 0.6201 | Valid Acc: 0.6046 | Valid mIoU: 0.1113 | train_time: 1272.7s | val_time: 36.5s
Best MS+Flip Acc: 0.6086 | Best MS+Flip mIoU: 0.1128
### b201xiaoli/seg-base-none-lr5e-5-60
Train Loss: 1.3830 | Train Acc: 0.6263 | Valid Acc: 0.6091 | Valid mIoU: 0.1212 | train_time: 1269.5s | val_time: 36.2s
Best MS+Flip Acc: 0.6139 | Best MS+Flip mIoU: 0.1222
### b201xiaoli/seg-base-none-lr7e-5-60
Train Loss: 1.3707 | Train Acc: 0.6272 | Valid Acc: 0.6108 | Valid mIoU: 0.1239 | train_time: 1270.7s | val_time: 36.3s
Best MS+Flip Acc: 0.6171 | Best MS+Flip mIoU: 0.1265

### kernel_id: jacksisi/seg-base-none-d-560
0.717464447	0.791200459	0.646057904	0.175850362	1274.446535	36.67424655
final_ms_flip_acc	final_ms_flip_miou
0.652945817	0.179681435

### kernel_id: jacksisi/seg-base-none-d-561
0.721958697	0.790141284	0.641946673	0.170735449	1272.759403	36.4370389
final_ms_flip_acc	final_ms_flip_miou
0.650624096	0.176641956

### kernel_id: sollasi/seg-base-abs-d-561
0.711122453	0.792918146	0.645678759	0.172757909	1282.235662	37.03127623
final_ms_flip_acc	final_ms_flip_miou
0.65581578	0.178854316
  
### kernel_id: pycjn666/seg-base-abs-d-560
0.690060616	0.798131824	0.649446189	0.179131553	1288.646698	38.18957925
final_ms_flip_acc	final_ms_flip_miou
0.657997966	0.184328616

### kernel_id: pycjn666/seg-base-rope-d-560
0.614112616	0.817926288	0.652353942	0.177536353	1346.161674	39.60817814
final_ms_flip_acc	final_ms_flip_miou
0.661408484	0.183233008

### kernel_id: sollasi/seg-base-rope-d-561
0.632707238	0.813044667	0.648901403	0.180312529	1347.441547	39.55108166
final_ms_flip_acc	final_ms_flip_miou
0.657597303	0.182482556
  
### kernel_id: cdong121/seg-base-colrow-d-561
2.762125492	0.777760744	0.65525347	0.180490613	1289.57689	37.02048373
final_ms_flip_acc	final_ms_flip_miou
0.663193285	0.184788272
  
### kernel_id: cdong121/seg-base-colrow-d-560
1.353509188	0.780703843	0.659483433	0.184741974	1282.478611	36.31470776
final_ms_flip_acc	final_ms_flip_miou
0.666284919	0.187009454
  
  
  

lr 5e-6 < lr 1e-5 < lr 5e-4; eta_min=1e-07>eta_min=5e-06
3e-5<3e-4 <2e-4 < 1e-4<5e-5<7e-5
scale_jitter=(1.0, 1.3) ms_scales=(0.9, 1.0, 1.15) good


## 20260126

### ssss7777/seg-base-none-d-462
0.641675651	0.173174977	1270.080759	36.24966121	130
final_ms_flip_acc	final_ms_flip_miou
0.650637865	0.177162245

### xuwenhui123/seg-base-none-d-463
0.643769264	0.171932027	1274.159467	36.52642822	130
final_ms_flip_acc	final_ms_flip_miou
0.651744783	0.176925763

### ssss7777/seg-base-abs-d-462
0.647718072	0.173013985	1278.032331	37.059834	130
final_ms_flip_acc	final_ms_flip_miou
0.656223953	0.179579079

### luanjing/seg-base-abs-d-463
0.646909654	0.1753553	1269.545855	36.35778284	130
final_ms_flip_acc	final_ms_flip_miou
0.654545426	0.178035215

### cycyxcy/seg-base-rope-d-462
0.645766675	0.174602062	1345.539452	39.54678535	130
final_ms_flip_acc	final_ms_flip_miou
0.655943811	0.17811574

### luanjing/seg-base-rope-d-463
0.649474025	0.176710337	1344.75557	39.53073621	130
final_ms_flip_acc	final_ms_flip_miou
0.658887625	0.182158306


### jjjerry12138/seg-base-colrow-d-462
valid_acc	valid_miou	train_time	val_time	epoch
0.657746851	0.182367831	1283.350677	36.40031314	130
final_ms_flip_acc	final_ms_flip_miou
0.664748371	0.185647383

### zhoujiahui0199/seg-base-colrow-d-463
0.662185669	0.189363912	1289.004605	36.83899689	130
final_ms_flip_acc	final_ms_flip_miou
0.669000089	0.193113744
