
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
namespace(model_type='dinov3', use_abs_pos_emb=True, use_rot_pos_emb=False, model_size='base', num_classes=150, batch_size=16, grad_accum_steps=1, train_img_size=336, eval_img_size=368, use_ms_flip_eval=False, scale_jitter=(1.0, 1.3), use_cat_max_ratio=True, cat_max_ratio=0.7, cat_max_ratio_tries=10, ms_scales=(0.9, 1.0, 1.15), eval_crop_mode='crop_or_pad', final_ms_flip_eval=True, lr=7e-05, lr_aux=1e-05, eta_min=1e-07, composite_lr=True, warmup_steps=500, weight_decay=0.01, epochs=130, overlap=0, start_epoch=0, seed=63, use_rc_loss=False, use_patch_position_loss=False, huber_beta=0.1, rc_alpha=70.0, seg_head='upernet', feature_layers=[2, 5, 8, 11], workers=2, color_jitter={'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.05}, color_jitter_prob=0.1, train=True, val=False, ckpt_path=None, lock=False, clip_value=1.0, output_dir='/kaggle/working', log_interval=300, csv_interval=3, show_peak_gpu_mem=True, compile_model=False, save_full_ckpt=True, resume_full_ckpt=True, resume_ckpt_path='/kaggle/input/seg-base-abs-d-363/ckpt/last.pth', resume_scheduler=True, resume_optimizer=True, resume_bs=True, total_run_time_hr=12.0, base_path='/kaggle/input/ade20k-dataset/ADEChallengeData2016', pos_type=None)
### ssss7777/seg-base-none-d-462
0.641675651	0.173174977	1270.080759	36.24966121	130
final_ms_flip_acc	final_ms_flip_miou
0.650637865	0.177162245

### xuwenhui123/seg-base-none-d-463
0.643769264	0.171932027	1274.159467	36.52642822	130
final_ms_flip_acc	final_ms_flip_miou
0.651744783	0.176925763

### xiaoluoalice/seg-base-none-d-550
0.719658196	0.790784121	0.642343462	0.169350088	1282.999801	37.3023417	130
final_ms_flip_acc	final_ms_flip_miou
0.651828766	0.173966348

### xiaoluoalice/seg-base-none-d-451
0.729467809	0.788225591	0.640669882	0.173220024	1275.896572	36.60493588	130
final_ms_flip_acc	final_ms_flip_miou
0.648902297	0.177117139

### keyongkk/seg-base-abs-d-451
0.71209991	0.792659581	0.647576809	0.172597885	1274.921215	36.49651408	130
final_ms_flip_acc	final_ms_flip_miou
0.655475378	0.176562101

### yinmengmeng/seg-base-abs-d-450
0.697800815	0.7961905	0.649634123	0.176099405	1296.798713	38.16317749	130
final_ms_flip_acc	final_ms_flip_miou
0.65649116	0.179158658

### ssss7777/seg-base-abs-d-462
0.647718072	0.173013985	1278.032331	37.059834	130
final_ms_flip_acc	final_ms_flip_miou
0.656223953	0.179579079

### luanjing/seg-base-abs-d-463
0.646909654	0.1753553	1269.545855	36.35778284	130
final_ms_flip_acc	final_ms_flip_miou
0.654545426	0.178035215

### xwj66666/seg-base-rope-d-450
0.634670198	0.812929571	0.646383822	0.167940184	1348.56264	39.84228659	130
final_ms_flip_acc	final_ms_flip_miou
0.656020105	0.173187241

### xwj66666/seg-base-rope-d-451
0.65290451	0.80789274	0.645679593	0.174261853	1350.596756	39.61256456	130
final_ms_flip_acc	final_ms_flip_miou
0.654930234	0.179377243

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

### jiangshuai0210/seg-base-colrow-d-450
1.151812553	0.781348646	0.661500692	0.187490165	1284.145727	36.51757836	130
final_ms_flip_acc	final_ms_flip_miou
0.668168783	0.189385042

### jiangshuai0210/seg-base-colrow-d-451
1.876859665	0.777040005	0.654198825	0.179953218	1286.630517	36.68391681	130
final_ms_flip_acc	final_ms_flip_miou
0.662066877	0.184522912


## 20260131

### xwj66666/seg-base-none-gpu-416
0.771407902	0.776618063	0.641697347	0.171990916	1262.336385	36.47632885	130
final_ms_flip_acc	final_ms_flip_miou
0.649674237	0.175417364
73.1s	27	368 0.641697347164154 0.17199091613292694
114.4s	28	384 0.6407278180122375 0.1698637455701828
158.9s	29	400 0.6399189829826355 0.17020906507968903
207.9s	30	416 0.6392571330070496 0.17101621627807617
### ohyeah00/seg-base-none-gpu-417
0.760007083	0.779763818	0.643711865	0.17262587	1272.581407	37.07170486	130
final_ms_flip_acc	final_ms_flip_miou
0.652065933	0.176799893
74.3s	27	368 0.6437118649482727 0.17262586951255798
115.2s	28	384 0.6437027454376221 0.17340485751628876
159.2s	29	400 0.6405048966407776 0.17198975384235382
207.8s	30	416 0.6401743292808533 0.17179661989212036
### permanentlove/seg-base-none-gpu-418
0.757934332	0.779818356	0.645438612	0.175202399	1271.548598	37.09197211	130
final_ms_flip_acc	final_ms_flip_miou
0.65279144	0.17866163
69.2s	27	368 0.6454386115074158 0.1752023994922638
110.4s	28	384 0.6425482630729675 0.1727963089942932
154.5s	29	400 0.6420108675956726 0.17362558841705322
203.4s	30	416 0.6412302851676941 0.17172552645206451
### xwj66666/seg-base-none-gpu-419
0.752100825	0.781532347	0.644161522	0.173152342	1272.860199	37.16211939	130
final_ms_flip_acc	final_ms_flip_miou
0.650033712	0.176016286
69.2s	27	368 0.6441615223884583 0.17315234243869781
110.3s	28	384 0.641832172870636 0.17160403728485107
154.7s	29	400 0.6410881876945496 0.17179331183433533
203.8s	30	416 0.6388469338417053 0.17167381942272186
### quepimao/seg-base-abs-gpu-416
0.740566015	0.784592688	0.646032095	0.173544854	1297.222202	41.55825353	130
final_ms_flip_acc	final_ms_flip_miou
0.655373633	0.179453567
eval_img_size	final_eval_acc	final_eval_miou
336	0.645537198	0.17267774
352	0.645098448	0.173824057
368	0.646032095	0.173544854
384	0.648205698	0.176735535
400	0.644886672	0.176376447
416	0.646648169	0.174656421
432	0.645908713	0.176069409
448	0.643178344	0.175065815
464	0.642701268	0.172385976
480	0.641707003	0.17220737
496	0.639888644	0.169276029
512	0.636648715	0.167927176


### ohyeah00/seg-base-abs-gpu-417
0.716862738	0.790187597	0.649845183	0.180700079	1266.853312	36.70409703	130
final_ms_flip_acc	final_ms_flip_miou
0.657883465	0.183769941
66.6s	27	368 0.6498451828956604 0.18070007860660553
107.7s	28	384 0.6495818495750427 0.17882893979549408
152.0s	29	400 0.6486495733261108 0.17910613119602203
201.0s	30	416 0.6495081782341003 0.17817018926143646
### permanentlove/seg-base-abs-gpu-418
0.72252214	0.789608538	0.65015763	0.177399442	1266.72654	36.82653975	130
final_ms_flip_acc	final_ms_flip_miou
0.657654762	0.18343313
78.4s	27	368 0.650157630443573 0.1773994415998459
119.3s	28	384 0.6502125263214111 0.17531776428222656
163.4s	29	400 0.6492185592651367 0.17795473337173462
212.0s	30	416 0.6490573883056641 0.17609067261219025
### quepimao/seg-base-abs-gpu-419
0.719109833	0.790305614	0.649613202	0.17706795	1268.222665	36.74227619	130
final_ms_flip_acc	final_ms_flip_miou
0.65716058	0.183632329
67.9s	27	368 0.6496132612228394 0.1770680993795395
108.8s	28	384 0.6486871838569641 0.17749977111816406
152.8s	29	400 0.6480860114097595 0.17879235744476318
201.5s	30	416 0.6490936279296875 0.17864878475666046

### wenyangtang/seg-base-rope-gpu-416
0.654816151	0.806656599	0.647436678	0.174145728	1346.495319	39.98198175	130
final_ms_flip_acc	final_ms_flip_miou
0.656979203	0.179595098
eval_img_size	final_eval_acc	final_eval_miou
336	0.646541715	0.174419224
352	0.648419201	0.17342405
368	0.647436678	0.174145728
384	0.648504019	0.173759237
400	0.648885489	0.175549954
416	0.648007214	0.17374894
432	0.646962225	0.17391777
448	0.64617449	0.173883855
464	0.645204723	0.171597168
480	0.643078744	0.171685115
496	0.640195549	0.168966934
512	0.640240729	0.1681339
### wenyangtang/seg-base-rope-gpu-417
0.66324234	0.804322839	0.653509498	0.176378146	1346.031107	39.67832541	130
final_ms_flip_acc	final_ms_flip_miou
0.662938178	0.1824774
78.7s	27	368 0.6535094976425171 0.17637814581394196
123.2s	28	384 0.6545390486717224 0.17898856103420258
171.2s	29	400 0.6534395813941956 0.17988143861293793
224.1s	30	416 0.654433012008667 0.17832063138484955
### xwj66666/seg-base-rope-gpu-418
0.67064333	0.802327573	0.647525966	0.173775047	1344.395942	39.90400624	130
final_ms_flip_acc	final_ms_flip_miou
0.656537175	0.177792385
71.4s	27	368 0.64752596616745 0.17377504706382751
115.7s	28	384 0.648006796836853 0.1735198199748993
163.5s	29	400 0.6481145620346069 0.1742410957813263
216.2s	30	416 0.64604252576828 0.17131274938583374
### rmoope/seg-base-rope-gpu-419
0.659712136	0.80483079	0.646821618	0.17517902	1344.699222	39.69354868	130
final_ms_flip_acc	final_ms_flip_miou
0.654203475	0.178806707
70.0s	27	368 0.6468216180801392 0.17517901957035065
114.5s	28	384 0.6477311849594116 0.1771027147769928
162.7s	29	400 0.6472485661506653 0.17497287690639496
216.0s	30	416 0.6444388628005981 0.1741115301847458

### uamadeus/seg-base-colrow-gpu-416
1.326491594	0.817602336	0.670458376	0.195836112	1269.993886	36.67495465	130
final_ms_flip_acc	final_ms_flip_miou
0.677997947	0.201430112

            368 0.6704583764076233 0.1958361119031906
114.1s	27	384 0.6701018214225769 0.19558151066303253
158.6s	28	400 0.6704660654067993 0.19599075615406036
208.0s	29	416 0.6701594591140747 0.19523237645626068
### qcx2333/seg-base-colrow-gpu-417
1.345056653	0.820343733	0.668736458	0.198825032	1281.436464	37.58112049	130
final_ms_flip_acc	final_ms_flip_miou
0.675410748	0.202321008

68.8s	26	368 0.668736457824707 0.19882503151893616
109.6s	27	384 0.668228268623352 0.19814065098762512
153.6s	28	400 0.668594241142273 0.1984853595495224
202.6s	29	416 0.6679514646530151 0.19726227223873138
### qcx2333/seg-base-colrow-gpu-418
1.303130984	0.826532662	0.669613123	0.200279906	1273.478718	37.19091797	130
final_ms_flip_acc	final_ms_flip_miou
0.677037776	0.20653227

69.6s	26	368 0.6696131229400635 0.20027990639209747
110.6s	27	384 0.6709550619125366 0.20175446569919586
154.7s	28	400 0.6687067151069641 0.19838882982730865
203.4s	29	416 0.6681739091873169 0.19983240962028503
### xx03071425/seg-base-colrow-gpu-419
1.272764921	0.822935462	0.669937313	0.197500363	1274.02575	37.08677053	130
final_ms_flip_acc	final_ms_flip_miou
0.677809298	0.200773984
68.6s	26	368 0.6699373126029968 0.19750036299228668
109.5s	27	384 0.6720248460769653 0.19845998287200928
153.6s	28	400 0.6706152558326721 0.19718335568904877
202.2s	29	416 0.6693446636199951 0.1951380968093872


## rc_alpha ablation
wsfa600: ra50 > ra60 > ra30 > ra70 > ra90 > ra70 > ra80 > ra300 > ra200 > ra500
wsfa600 > wsfa300
ampere888/seg-base-colrow-ra70-wsfa600-16
Best MS+Flip Acc: 0.6296 | Best MS+Flip mIoU: 0.1353

ampere888/seg-base-colrow-ra80-wsfa600-16
Best MS+Flip Acc: 0.6295 | Best MS+Flip mIoU: 0.1352

asdsad0000/seg-base-colrow-ra90-wsfa600-16
Best MS+Flip Acc: 0.6336 | Best MS+Flip mIoU: 0.1380

cshlhs/seg-base-colrow-ra100-wsfa600-16
Best MS+Flip Acc: 0.6301 | Best MS+Flip mIoU: 0.1356

cshlhs/seg-base-colrow-ra200-wsfa600-16
Best MS+Flip Acc: 0.6239 | Best MS+Flip mIoU: 0.1273

cycyxcy/seg-base-colrow-ra300-wsfa600-16
Best MS+Flip Acc: 0.6201 | Best MS+Flip mIoU: 0.1274

cycyxcy/seg-base-colrow-ra500-wsfa600-16
Best MS+Flip Acc: 0.6151 | Best MS+Flip mIoU: 0.1229

dddd110/seg-base-colrow-ra50-wsfa600-16
Best MS+Flip Acc: 0.6335 | Best MS+Flip mIoU: 0.1431

dddd110/seg-base-colrow-ra60-wsfa600-16
Best MS+Flip Acc: 0.6339 | Best MS+Flip mIoU: 0.1411

denghaimeng/seg-base-colrow-ra30-wsfa600-16
Best MS+Flip Acc: 0.6308 | Best MS+Flip mIoU: 0.1398

houwen/seg-base-colrow-ra70-wsfa600-16
Best MS+Flip Acc: 0.6344 | Best MS+Flip mIoU: 0.1391

huoqiuxia/seg-base-colrow-ra70-wsfa300-16
Best MS+Flip Acc: 0.6338 | Best MS+Flip mIoU: 0.1382