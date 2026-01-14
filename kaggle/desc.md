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
14. if cfg.resume_full_ckpt and cfg.resume_infer both true, infer following in cfg: task, method, model_size, seed. the name of kernel_sources or dataset_sources is the id in previous json_f;
15. please generate a kaggle/kernel.py, it can be provided with args id, kernel_id, output and delete. if arg id is not provided, use the id in kaggle/
  config.yaml; if kernel_id not given, use id of json_f;  if kernel_id don't contain '/', make it as id/kernel_id; retrieve token with id from kaggle/tokens.yaml; set env KAGGLE_API_TOKEN. if flag delete is given, the run "kaggle kernels delete -y kernel_id"; if flag output is given, the run "kaggle kernels output kernel_id"; if both delete and output not given, "kaggle kernels status kernel_id".
16.  modify kernel.py: 1. to accept a --dry arg, if dry, don't actually issue kaggle command, just report it; 2. to accept a .md file as kernel_id: if arg kernel_id is a .md file name, read and process the lines one by one; 3. if a line in the .md file don't contain '/', it is not a valid kernel_id, skip it; 4. if --delete is true, if a line not started with a -, skip it; 5. trim the space or '?' in the start or end of the line; 6. issue the kaggle command for the kernel_id of this line with correct options and not dry