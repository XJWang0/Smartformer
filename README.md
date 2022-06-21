# Code of Smartformer



## Requirement

- python >= 3.6
- torch >= 1.4.0
- tensorly==0.5.0
- torchsummary
- einops
- sklearn

## Usage

```python
# EEG-Transformer
python EEG-Transformer/smartformer.py
# Autoformer
bash Autoformer/scripts/Weather_script/Smart_Autoformer.sh
# DEformer
bash deformer/smartDE_binarized_MNIST.sh
```



If you want more details about EEG-Transformer, Autoformer and DEformer, please see [**Transformer-based Spatial-Temporal Feature
Learning for EEG Decoding**][https://arxiv.org/abs/2106.11170], [**Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting**][https://arxiv.org/abs/2106.13008] and [**The DEformer: An Order-Agnostic Distribution Estimating Transformer**][https://arxiv.org/abs/2106.06989]. And the code [**EEG-Transformer**][ https://github.com/anranknight/EEG-Transformer], [**Autoformer**][https://github.com/thuml/Autoformer] and [**DEformer**][https://github.com/airalcorn2/deformer].