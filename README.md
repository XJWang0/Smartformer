# **Smartformer: An Intelligent Model Compression Framework for Transformer**

Transformer equipped with large amounts of parameters, has achieved some outstanding performance in natural language processing, computer visions, etc. Large amouts of parameters  result in the storage burden under a resource-limited setting . Deploying these parameters is a challenge in a resource-limited setting, and over-parameterization inevitably introduces local optima that hinders the convergence in training. Literature works has proposed several ways to relieve the over-parameterization issue; however, none of them has focused on addressing it simultaneously considering the following three objectives: (1) preserve the model architecture, (2) maintain the model capacity (performance on a given task), (3) and reduce the model complexity (number of parameters) to improve its convergence property. In this study, we propose a novel Transformer model compression framework, Smartformer, to tackle these three concerns by incorporating reinforcement learning and CP-decomposition method. The experiment results demonstrate the high effectiveness of the Smartformer on existing models and datasets in terms of model performance improvement and model parameter size reduction.

## Requirement

- python >= 3.6
- torch >= 1.4.0
- tensorly==0.5.0
- torchsummary
- einops
- sklearn

# cp-trans

This package contains the implementation of Actor-Critic, Encoder-Net, Multi-Head Attention using CP-Decomposition and other related functions.

## Usage

We have set up 3 experiments, namely EEG-Transformer, Autoformer and DEformer. You can train by running the following commands. Please refer to our paper for the experiment results. You can view relevant project files for more parameter settings.

```python
# EEG-Transformer
python EEG-Transformer/smartformer.py
# Autoformer
bash Autoformer/scripts/Weather_script/Smart_Autoformer.sh
# DEformer
bash deformer/smartDE_binarized_MNIST.sh
```



If you want more details about EEG-Transformer, Autoformer and DEformer, please see [**Transformer-based Spatial-Temporal Feature
Learning for EEG Decoding**](https://arxiv.org/abs/2106.11170), [**Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting**](https://arxiv.org/abs/2106.13008) and [**The DEformer: An Order-Agnostic Distribution Estimating Transformer**](https://arxiv.org/abs/2106.06989). And the code [**EEG-Transformer**](https://github.com/anranknight/EEG-Transformer), [**Autoformer**](https://github.com/thuml/Autoformer) and [**DEformer**](https://github.com/airalcorn2/deformer).
