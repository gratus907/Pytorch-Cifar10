| Model            | Optimizer    | Hyperparams                                                  | Result(Accuracy) |
| ---------------- | ------------ | ------------------------------------------------------------ | ---------------- |
| LeNet-Naive      | SGD-Momentum | lr=2e-2, momentum=0.95                                       | 51.32%           |
| AlexNet          | SGD-Momentum | lr=1e-2, wd=1e-5, momentum=0.95, LR-reduceonplateu(10, 0.1)  | 90.82%           |
| VGG16            | SGD-Momentum | lr=1e-2, wd=0.0005, momentum=0.9, LR-reduceonplateu(10, 0.1) | 91.72%(50epoch)  |
| GoogLeNet        | SGD-Momentum | lr=1e-2, wd=0.0005, momentum=0.9, LR-reduceonplateu(10, 0.1) | 93.57%(50epoch)  |
|                  |              |                                                              |                  |