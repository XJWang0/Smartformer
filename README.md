# **Smartformer: An Intelligent Model Compression Framework for Transformer**

Transformer, as one of the latest popular deep neural networks (DNNs), has achieved outstanding performance in various areas. However, this model usually requires massive amounts of parameters to fit. Over-parameterization not only brings storage challenges in a resource-limited setting but also inevitably results in the model over-fitting. Even though literature works introduced several ways to reduce the parameter size of Transformers, none of them have focused on addressing the over-parameterization issue by concurrently considering the following three objectives: preserving the model architecture, maintaining the model performance, and reducing the model complexity (number of parameters) to improve its convergence property. In this study, we propose an intelligent model compression framework, Smartformer, by incorporating reinforcement learning and CP-decomposition techniques. This framework has two advantages: 1) from the perspective of convenience, researchers can directly exploit this framework to train their designed Transformer models as usual without fine-tuning the original model training hyperparameters; 2) from the perspective of effectiveness, researchers can simultaneously fulfill the aforementioned three objectives by using this framework. We conduct two basic experiments and three practical experiments to demonstrate the proposed framework using various datasets and existing Transformer models.
The results demonstrate that the Smartformer is able to improve the existing Transformers regarding the model performance and model complexity by intelligently incorporating the model decomposition and compression in the training process.

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

We have set up 3 experiments, namely EEG-Transformer, Autoformer and DEformer. First, you need to set cp_ trans copy to the project you want to run. Then, you can train by running the following commands. For EEG-Transformer, you can get the data and label from [here](), then run ```getData.m``` using matlab to process. Please refer to our [paper](https://www.researchsquare.com/article/rs-1780688/v1) for the experiment results. You can view relevant project files for more parameter settings.

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
