1. please analysis kaggle/kernel.py for query kernel status with kernel id; and analysis kaggle/process_kaggle.py for resume_infer from a kernel resume_source, and how to submit (push) a kaggle kernel;
2. write a kaggle/auto_resume.py, to run with kaggle/config_kernel.yaml (kcfg). Do the following:
3. First, kaggle/auto_resume.py sleep for kcfg.sleep_time_hr hours;
4. for every kcfg.poll_interval_minutes minutes, poll the kaggle status of each kerel_id in notebooks of each item in running_nodes;
5. do nothing if kernel is running; report the error msg if errors;
6. if status of a kernel_id is finished, run_id += 1, if resumed_from of current notebook is not None, move it to the list history_ids; current kernel_id move to resumed_from;
7. if run_id==total_runs, move the current notebook block to kcfg.finished_notebooks (remove start_time, run_id, resumed_from fields); 
8. else if run_i < total_runs, set left_gpu_time of current node as: left_gpu_time = left_gpu_time - now - start_time (in hours);
10. if left_gpu_time<=0: get a id from kcfg.available_ids (and remove it from kcfg.available_ids), initialize a new item in running nodes: id is the id, left_gpu_time=30; move current item in notebooks to the notebooks of the new node; if current running node has no other notebooks, remove it from running_nodes, and put its id in exhausted_ids;
11. Use logic like resume_infer in kaggle/process_kaggle.py to infer cfg fields from resumed_from id (run_id used as suffix in cfg), and put the resulted kernel id in the field kernel_id of kcfg.
12. submit the new kernel to kaggle like that in kaggle/process_kaggle.py;
13. existing methods in kaggle/process_kaggle.py and kaggle/kernel.py can be imported and reused, the methods can be refactored for better reuse if necessary.