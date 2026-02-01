
### guyuefangyuan6666/seg-base-none-tpu8c-50
7908.3s	517	2026-01-31 06:21:39,515 - INFO - Epoch 88/130 Summary:
7908.3s	518	2026-01-31 06:21:39,516 - INFO - Step 9240 Summary:
7908.3s	519	2026-01-31 06:21:39,516 - INFO -   Train Loss: 1.4508 | Train Acc: 0.6088 | Valid Acc: 0.5996 | Valid mIoU: 0.1153 | train_time: 149.4s | val_time: 9.9s


# lr ablation

  - kernel_id: robinrainy/seg-base-abs-lr7-50
  1.284172297	0.645126998	0.613853395	0.130906329	82.69305968	7.371065855	97
nan
 
  - kernel_id: xulijuan/seg-base-abs-lr5-50
1.264847875	0.649813235	0.621409059	0.132875949	82.45865297	7.463193655	98
nan

  - kernel_id: guyuefangyuan6666/seg-base-abs-lr3-50
1.253396034	0.653880477	0.627223492	0.133852154	84.82031035	7.598225594	129
