#


## 1. How to run

Step 1:
```bash
conda create -n segmentation python=3.9 -y
```

Step 2: 
```bash
conda activate segmentation
```

Step 3:
```bash
pip install --no-cache-dir -r requirements.txt
```
## 2. Results


| Condition          | Backbone        | Classification F1 Score | Lung segmentation F1 Score | Lung segmentation IoU | Lung segmentation Dice | Infection segmentation F1 Score | Infection segmentation IoU | Infection segmentation Dice | Mean F1 |
|--------------------|----------------|-----------------------|--------------------------|---------------------|----------------------|--------------------------------|---------------------------|---------------------------|---------|
| w/o processing      | Resnet50        | 91.31                 | 96.52                    | 93.29               | 96.48                | 81.59                          | 68.64                     | 78.10                     | 89.80   |
|                    | Mobilenetv3     | 90.95                 | 93.99                    | 92.15               | 95.83                | 83.99                          | 73.19                     | 82.71                     | 89.64   |
|                    | Densenet121     | 93.44                 | 96.76                    | 93.75               | 96.73                | 86.39                          | 76.55                     | 85.07                     | 92.19   |
|                    | Inceptionv4     | 91.65                 | 95.17                    | 90.81               | 95.09                | 80.65                          | 68.77                     | 79.21                     | 89.15   |
|                    | efficientnet-b4 | 92.62                 | 96.49                    | 93.23               | 96.45                | 85.39                          | 75.56                     | 84.36                     | 91.50   |
|                    |                |                         |                          |                     |                       |                                |                            |                            |         |
| w post processing  | Resnet50        | 91.72                 | 96.51                    | 93.27               | 96.46                | 82.50                          | 69.97                     | 78.17                     | 90.24   |
|                    | Mobilenetv3     | 91.48                 | 95.88                    | 92.11               | 95.79                | 85.31                          | 75.38                     | 83.69                     | 90.89   |
|                    | Densenet121     | 94.02                 | 96.74                    | 93.72               | 96.70                | 87.34                          | 78.18                     | 85.65                     | 92.70   |
|                    | Inceptionv4     | 92.09                 | 95.14                    | 90.77               | 95.05                | 82.21                          | 71.30                     | 80.39                     | 89.81   |
|                    | efficientnet-b4 | 93.05                 | 96.47                    | 93.20               | 96.42                | 86.62                          | 77.53                     | 85.27                     | 92.04   |



(16, 3, 256, 256)
(16, 256, 768)
(16, 256, 768)
(16, 256, 768)
(16, 256, 768)
(16, 256, 768)

to this size 
(16, 3, 256, 256)
(16, 64, 128, 128)
(16, 256, 64, 64)
(16, 512, 32, 32)
(16, 1024, 16, 16)
(16, 2048, 8, 8)


(16, 3, 256, 256)
(16, 64, 128, 128)
(16, 192, 64, 64)
(16, 384, 32, 32)
(16, 1024, 16, 16)
(16, 1536, 8, 8)

import numpy as np
list_of_tensors_cpu = [tensor.detach().cpu().numpy() for tensor in features]
print(len(list_of_tensors_cpu))
for tensor in list_of_tensors_cpu:
    print(tensor.shape)