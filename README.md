[RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)

### Installation

```
conda create -n PyTorch python=3.8
conda activate PyTorch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
pip install opencv-python==4.5.5.64
pip install pyyaml
pip install timm
pip install tqdm
```

### Note

* The default training configuration is for `RepVGG-A0`
* The test results including accuracy, params and FLOP are obtained by using fused model

### Parameters and FLOPS

```
Number of parameters: 8.309M
Time per operator type:
        13.9458 ms.    93.5989%. Conv
       0.606509 ms.    4.07065%. Relu
       0.332151 ms.    2.22927%. FC
      0.0136558 ms.  0.0916524%. AveragePool
     0.00142615 ms. 0.00957176%. Flatten
        14.8996 ms in Total
FLOP per operator type:
        2.72034 GFLOP.    99.9059%. Conv
       0.002561 GFLOP.   0.094054%. FC
              0 GFLOP.          0%. Relu
         2.7229 GFLOP in Total
Feature Memory Read per operator type:
        35.6399 MB.    74.3363%. Conv
        7.17517 MB.    14.9657%. Relu
        5.12912 MB.    10.6981%. FC
        47.9442 MB in Total
Feature Memory Written per operator type:
        7.17517 MB.    49.9861%. Conv
        7.17517 MB.    49.9861%. Relu
          0.004 MB.  0.0278661%. FC
        14.3543 MB in Total
Parameter Memory per operator type:
        28.0956 MB.    84.5753%. Conv
          5.124 MB.    15.4247%. FC
              0 MB.          0%. Relu
        33.2196 MB in Total
```

### Train

* Configure your `IMAGENET` dataset path in `main.py` for training
* Run `bash main.sh $ --train` for training, `$` is number of GPUs

### Test

* Configure your `IMAGENET` path in `main.py` for testing
* Run `python main.py --test` for testing

### Results

|  Version   | Epochs | Top-1 Acc | Top-5 Acc | Params (M) | FLOP (G) |                                                                       Download |
|:----------:|:------:|----------:|----------:|-----------:|---------:|-------------------------------------------------------------------------------:|
| RepVGG-A0  |  120   |         - |         - |      8.309 |    1.362 |                                                                              - |
| RepVGG-A0* |  120   |      72.4 |      90.5 |      8.309 |    1.362 | [model](https://github.com/jahongir7174/RepVGG/releases/download/v0.0.1/A0.pt) |
| RepVGG-A1* |  120   |      74.5 |      91.8 |     12.790 |    2.364 | [model](https://github.com/jahongir7174/RepVGG/releases/download/v0.0.1/A1.pt) |
| RepVGG-A2* |  120   |      76.5 |      93.0 |     25.500 |    5.117 | [model](https://github.com/jahongir7174/RepVGG/releases/download/v0.0.1/A2.pt) |
| RepVGG-B0* |  120   |      75.1 |      92.4 |     14.339 |    3.058 | [model](https://github.com/jahongir7174/RepVGG/releases/download/v0.0.1/B0.pt) |
| RepVGG-B1* |  120   |      78.3 |      94.1 |     51.829 |   11.816 | [model](https://github.com/jahongir7174/RepVGG/releases/download/v0.0.1/B1.pt) |
| RepVGG-B2* |  120   |      78.8 |      94.4 |     80.315 |   18.377 | [model](https://github.com/jahongir7174/RepVGG/releases/download/v0.0.1/B2.pt) |

* `*` means that weights are ported from original repo, see reference

#### Reference

* https://github.com/DingXiaoH/RepVGG
