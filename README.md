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
## 2. Results (CNN)


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

## 2. Results (Transformer)

|                  | Backbone | Classification F1 score | Lung segmentation F1 score | Lung segmentation IoU | Lung segmentation Dice | Infection segmentation F1 score | Infection segmentation IoU | Infection segmentation Dice | Mean F1 score |
|------------------|----------|--------------------------|---------------------------|-----------------------|------------------------|--------------------------------|-----------------------------|------------------------------|---------------|
| w/o processing   | mit_b0   | 86.45                    | 95.85                     | 92.05                 | 95.73                  | 83.86                          | 72.42                       | 78.23                        | 88.72         |
|                  | mit_b1   | 87.99                    | 96.19                     | 92.67                 | 96.10                  | 85.72                          | 75.48                       | 81.09                        | 89.97         |
|                  | mit_b2   | 91.68                    | 96.19                     | 92.68                 | 96.10                  | 87.30                          | 78.49                       | 84.08                        | 91.72         |
|                  | mit_b3   | 91.34                    | 96.20                     | 92.71                 | 96.11                  | 86.62                          | 77.25                       | 82.92                        | 91.38         |
|                  |          |                          |                           |                       |                        |                                |                             |                              |               |
| w post processing | mit_b0   | 86.45                    | 95.85                     | 92.07                 | 95.74                  | 84.10                          | 72.85                       | 78.65                        | 88.8          |
|                  | mit_b1   | 87.99                    | 96.19                     | 92.68                 | 96.11                  | 86.03                          | 75.93                       | 81.54                        | 90.07         |
|                  | mit_b2   | 91.68                    | 96.19                     | 92.70                 | 96.11                  | 87.74                          | 79.02                       | 84.59                        | 91.87         |
|                  | mit_b3   | 91.34                    | 96.21                     | 92.73                 | 96.13                  | 86.89                          | 77.60                       | 83.26                        | 91.48         |
