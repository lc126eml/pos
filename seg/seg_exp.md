
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

lr 5e-6 < lr 1e-5 < lr 5e-4; eta_min=1e-07>eta_min=5e-06
scale_jitter=(1.0, 1.3) ms_scales=(0.9, 1.0, 1.15) good
## use_cat_max_ratio=True
