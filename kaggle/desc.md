generate process_kaggle.py in /home/liucong/codes/pos/kaggle:
1. read kaggle/config.yaml as a dict, e.g., cfg;
2. decide python file (py_file) based on cfg.task: if task is seg, python file is seg/dinov3_seg_kaggle.py; it is dinov3_reg_dynamic.py if cls;
3. modify the args of py_file and kaggle/kernel-metadata.json based on the cfg with following rules;
4. modify kaggle/kernel-metadata.json (json_f), all values in this file is of string type: update value of is_private in json_f with cfg.is_private (string);
5. set id of json_f as: {cfg.id}/{cfg.task}-{cfg.model_size}-{cfg.method}{cfg.seed}
6. set title of json_f as: {cfg.task} {cfg.model_size} {cfg.method}{cfg.seed}, i.e., it is the second part of id but replace '-' with space;
7. dataset_sources of json_f is a list: it must contain "liucong12601/timm-repos"; if cfg.task is seg, add "awsaf49/ade20k-dataset", if cls add "ambityga/imagenet100"; if cfg.dataset_sources is not None, add it;
8. add cfg.kernel_sources to kernel_sources of json_f, it is a list of string too;
9. modify the args of py_file, set its seed with cfg.seed; set model_size, resume_full_ckpt from that of cfg;