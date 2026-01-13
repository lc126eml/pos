generate process_kaggle.py in /lc/code/pos/kaggle:
1. read kaggle/config.yaml as a dict, e.g., cfg;
2. decide python file (py_file) based on cfg.task: if task is seg, python file is seg/dinov3_seg_kaggle.py; it is dinov3_reg_dynamic.py if cls;
3. modify the args of py_file and kaggle/kernel-metadata.json based on the cfg with following rules;
4. modify kaggle/kernel-metadata.json (json_f), all values in this file is of string type: update value of is_private in json_f with cfg.is_private (string);
5. set id of json_f as: {cfg.id}/{cfg.task}-{cfg.model_size}-{cfg.method}{cfg.suffix}{cfg.seed}
6. set title of json_f as: {cfg.task} {cfg.model_size} {cfg.method}{cfg.suffix}{cfg.seed}, i.e., it is the second part of id but replace '-' with space;
7. dataset_sources of json_f is a list: it must contain "liucong12601/timm-repos"; if cfg.task is seg, add "awsaf49/ade20k-dataset", if cls add "ambityga/imagenet100"; if cfg.dataset_sources is not None, add it;
8. add cfg.kernel_sources to kernel_sources of json_f, it is a list of string too;
9. modify the args of py_file, set its seed with cfg.seed; set model_size, resume_full_ckpt from that of cfg;
10. if resume_full_ckpt is true: and if resume_source is kernel, source_name as second part (after '/') of cfg.kernel_sources; if is resume_source is dataset, source_name as second part (after '/') of cfg.dataset_sources. then make arg resume_ckpt_path of py_file as "/kaggle/input/{}/ckpt/last.pth";
11. cfg.method is rope, set use_rot_pos_emb to true, but use_abs_pos_emb and use_rc_loss to False; if cfg.method is abs, only use_abs_pos_emb True; if colrow, only use_rc_loss True; if none, All False.
12. for items in cfg.simple, set the values of args in py_file with the key.
13. set cfg.pos_type to args of py_file. if cfg.pos_type is not None, set use_rot_pos_emb, use_abs_pos_emb, use_rc_loss, dynamic_img_size, use_patch_position_loss, val   to False. id of json_f as:  {cfg.id}/{cfg.task}-{cfg.model_size}-{cfg.pos_type}{cfg.suffix}{cfg.seed}, and title too