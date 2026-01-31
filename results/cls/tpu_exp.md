# cls

### sinayliu/cls-base-none-lr1e-4-50
Train Loss: 0.7185 | Train Acc: 0.8000 | Valid Acc: 0.5774 | train_time: 99.3s | val_time: 4.3s Best Accuracy: 0.5806
### pycjn666/cls-base-none-lr2e-4-50
Train Loss: 1.8828 | Train Acc: 0.5121 | Valid Acc: 0.4538 | train_time: 94.5s | val_time: 4.3s Best Accuracy: 0.4578
### liucong12601/cls-base-none-lr8e-5-50
Train Loss: 0.5907 | Train Acc: 0.8361 | Valid Acc: 0.5998 | train_time: 97.0s | val_time: 4.9s Best Accuracy: 0.6022
### jjjerry12138/cls-base-none-lr5e-5-50
Train Loss: 0.8806 | Train Acc: 0.7542 | Valid Acc: 0.5924 | train_time: 96.9s | val_time: 4.2s Best Accuracy: 0.5950
### zjl001/cls-base-none-lr3e-5-50
Train Loss: 1.4000 | Train Acc: 0.6213 | Valid Acc: 0.5474 | train_time: 95.6s | val_time: 4.4s Best Accuracy: 0.5486


### du55148/cls-base-none-tpu-50
0.611393154	0.830973685	0.604399979	97.64794803	4.122288227	130

### asdsad0000/cls-base-none-tpu-51
0.617178142	0.829483747	0.604399979	94.89224839	4.034758329	130

### ohyeah00/cls-base-none-tpu-52
0.612664878	0.830973685	0.600799978	92.84575415	4.199072123	130

### cdong121/cls-base-none-tpu-53
0.611410677	0.832440436	0.609399974	93.63991523	4.16623354	130

### liucong12601/cls-base-rope-d-50
0.684410036	0.808547437	0.645799994	94.76681852	4.197659016	130

### liucong12601/cls-base-rope-d-51
0.728503227	0.794474185	0.640199959	96.40247488	4.087512255	130

### liucong12601/cls-base-rope-tpu-52
0.671781898	0.811751187	0.637199998	103.1516693	4.426205635	130

### zjl001/cls-base-rope-tpu-53 no eval
0.703108132	0.803498685	0.637799978	100.8530593	4.391128778	130



### sinayliu/cls-base-abs-d-50
0.539301336	0.850435436	0.625400007	93.97559118	4.103694439	130

### wenyangtang/cls-base-abs-tpu-51
0.586816013	0.836423874	0.618399978	97.15248823	4.129728317	130


### xulijuan/cls-base-abs-tpu-52
0.599412441	0.833783686	0.625199974	116.3930092	4.555068493	130


### jacksisi/cls-base-abs-tpu-53
0.537404418	0.850057185	0.63440001	99.09350777	4.262759686	130


### yuanhahah/cls-base-relpos-d-50
0.522795796	0.855515122	0.638599992	98.03561115	4.177647114	130

### cycyxcy/cls-base-alibi-d-50
0.568491817	0.84163481	0.628399968	94.14329076	4.277119875	130



# mres cls
### cshlhs/cls-base-none-is224192288-50
0.940867901	0.736342847	0.596599996	91.61576748	4.223912716	130

### jacksisi/cls-base-none-mres-50
0.940867901	0.736342847	0.596599996	88.09301829	4.29629612	130

### wenyangtang/cls-base-rope-mres-50
0.93706125	0.736543715	0.625999987	89.5775888	4.116618395	130

### liucong126/cls-base-abs-mres-50
0.763247907	0.784706771	0.619599998	91.4333396	4.219184637	130

## rc alpha ablation
namespace(pos_type=None, dynamic_img_size=True, model_type='dinov3', use_abs_pos_emb=False, use_rot_pos_emb=False, model_size='base', num_classes=100, patch_size=16, **batch_size=64**, img_sizes=[224], val_img_sizes=[160, 176, 192, 208, 224, 256, 272, 288, 320, 336, 352, 368, 384, 400, 416], **lr=0.00028,** lr_aux=4e-05, eta_min=0.0, weight_decay=0.01, epochs=130, overlap=0, pretrained=None, seed=50, use_patch_position_loss=False, use_rc_loss=True, rc_alpha=200, warmup_steps_for_aux=1, workers=0, re_prob=0.0, train=True, val=True, tpu_size_schedule='epoch', tpu_size_hold_batches=0, tpu_workers=0, tpu_threads=1, ckpt_path=None, lock=False, save_full_ckpt=False, resume_full_ckpt=False, resume_ckpt_path=None, resume_scheduler=True, resume_optimizer=True, resume_bs=True, composite_lr=True, warmup_steps=3000, clip_value=1.0, log_interval=100, csv_interval=1, show_peak_gpu_mem=False, compile_model=False, debug_xla=True, log_all_ranks=False, total_run_time_hr=9.0, root_dir='/kaggle/working')
### xulijuan/cls-base-colrow-ra200-50
Train Loss: 5.6337 | Aux Loss: 0.0221 | Base Loss: 1.2189 | Train Acc: 0.6670 | Valid Acc: 0.6198 | train_time: 91.1s | val_time: 4.0s
Best Accuracy: 0.6214
empty csv

### jacksisi/cls-base-colrow-tpu-50 bad
lr=0.00028, lr_aux=4e-05, eta_min=0.0 use_rc_loss=True, rc_alpha=600.0

### ### sinayliu/cls-base-colrow-ra300-50
lr=0.00028, lr_aux=4e-05, eta_min=0.0
Train Loss: 7.8943 | Aux Loss: 0.0212 | Base Loss: 1.5333 | Train Acc: 0.5891 | Valid Acc: 0.5678 | train_time: 94.3s | val_time: 4.1s
epoch_val_acc=0.5679999589920044

###  kernel_id: liucong126/cls-base-colrow-ra100-50
Train Loss: 3.3367 | Aux Loss: 0.0257 | Base Loss: 0.7626 | Train Acc: 0.7856 | Valid Acc: 0.6472 | train_time: 95.6s | val_time: 4.2s

###  kernel_id: zjl001/cls-base-colrow-mres-50 bad
img_sizes=[224, 192, 288], lr=0.00028, lr_aux=4e-05, eta_min=0.0 use_rc_loss=True, rc_alpha=600.0
###  kernel_id: ampere888/cls-base-colrow-ra400-50 bad