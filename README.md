# Smartformer: An Intelligent Model Compression Framework for Transformer
Transformer, as one of the latest popular deep neural networks (DNNs), has achieved outstanding performance in various areas. However, this model usually requires massive amounts of parameters to fit. Over-parameterization not only brings storage challenges in a resource-limited setting but also inevitably results in the model over-fitting. Even though literature works introduced several ways to reduce the parameter size of Transformers, none of them addressed the issues caused by over-parameterization by concurrently achieving the following three objectives: preserving the model architecture, maintaining the model performance, and reducing the model complexity (number of parameters). In this study, we propose an intelligent model compression framework, Smartformer, by incorporating reinforcement learning and CP-decomposition techniques. This framework has two advantages: 1) from the perspective of convenience, researchers can directly exploit this framework to train their designed Transformers without changing the hyperparameters for the regular training process; 2) from the perspective of effectiveness, researchers can simultaneously fulfill the aforementioned three objectives by using this framework. In the experiment, we apply Smartformer and five baseline methods to three existing Transformer models for model compression. The results demonstrate that our proposed Smartformer is the only one method that consistently generates the compressed model on various scenarios by satisfying the three objectives.



# Requirement

- python >= 3.6
- torch >= 1.4.0
- torchvision
- tensorly == 0.5.0
- torchsummary
- einops
- sklearn
- timm
- ptflops
- fvcore



# Uasge

We have set up 3 experiments, namely *EEG-Transformer*, *Autoformer* and *Compact-Transformer*. You can train by running the following commands.  You can view relevant project files for more parameter settings.

```python
# EEG-Transformer
## benchmarks
bash EEG/run_benchmarks.sh
## smart_version
bash EEG/run_smart.sh

# Autoformer
## benchmarks
bash Autoformer/run_benchmarks.sh
## smart_version
bash Autoformer/run_smart.sh

# Compact-Transformer
## Cifar10
### benchmarks
bash Compact-Transformers/cifar10_benchmarks.sh
### smart_version
bash Compact-Transformers/cifar10_smart.sh

## Cifar100
### benchmarks
bash Compact-Transformers/cifar100_benchmarks.sh
### smart_version
bash Compact-Transformers/cifar100_smart.sh
```

For **Cifar10** dataset, you can run the following commands to obtain, and **Cifar100** is also the same:

```
wget https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz
gunzip cifar10.tgz
tar -xvf cifar10.tar
```



If you want more details about EEG-Transformer, Autoformer and Compact-Transformer, please see [**Transformer-based Spatial-Temporal Feature Learning for EEG Decoding**](https://arxiv.org/abs/2106.11170), [**Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting**](https://arxiv.org/abs/2106.13008) and [**Escaping the Big Data Paradigm with Compact Transformers**](https://arxiv.org/abs/2104.05704). And the code [**EEG-Transformer**](https://github.com/anranknight/EEG-Transformer), [**Autoformer**](https://github.com/thuml/Autoformer) and [**Compact-Transformer**](https://github.com/SHI-Labs/Compact-Transformers).