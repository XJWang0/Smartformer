# Smartformer: An Intelligent Transformer Compression Framework for Time-Series Modeling
Transformer, as one of the cutting-edge deep neural networks (DNNs), has achieved outstanding performance in time-series data analysis. However, this model usually requires massive amounts of parameters to fit. Over-parameterization not only brings storage challenges in a resource-limited setting but also inevitably results in the model over-fitting. Even though literature works introduced several ways to reduce the parameter size of Transformers, none of them addressed this over-parameterized issue  by concurrently achieving the following three objectives: preserving the model architecture, maintaining the model performance, and reducing the model complexity (number of parameters). In this study, we propose an intelligent model compression framework, Smartformer, by incorporating reinforcement learning and CP-decomposition techniques. This framework has two advantages: 1) from the perspective of convenience, researchers can directly exploit this framework to train their designed Transformers without changing the hyperparameters for the regular training process; 2) from the perspective of effectiveness, researchers can simultaneously fulfill the aforementioned three objectives by using this framework. In the experiment, we apply Smartformer and five baseline methods to two existing time-series Transformer models for model compression. The results demonstrate that our proposed Smartformer is the only method that consistently generates the compressed model on various scenarios by satisfying the three objectives. In particular, the Smartformer can mitigate the overfitting issue and thus improve the accuracy of the existing time-series models in all scenarios.

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
We have set up 2 experiments, namely EEG-Transformer and Autoformer. You can train by running the following commands.  You can view relevant project files for more parameter settings.

`````python
# EEG-Transformer
## benchmarks
bash EEG/run_benchmarks.sh
## smartformer
bash EEG/run_smart1.sh
## the different ranks R
bash EEG/run_different_R1.sh
## the sensitivity
bash EEG/run_sensitivity1.sh

# Autoformer
## benchmarks
bash Autoformer/run_benchmarks.sh
## smartformer
bash Autoformer/run_smart2.sh
## the different ranks R
bash Autoformer/run_different_R2.sh
## the sensitivity
bash Autoformer/run_sensitivity2.sh
`````

If you want more details about EEG-Transformer and Autoformer, please see [**Transformer-based Spatial-Temporal Feature Learning for EEG Decoding**](https://arxiv.org/abs/2106.11170), [**Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting**](https://arxiv.org/abs/2106.13008). And the code [**EEG-Transformer**](https://github.com/anranknight/EEG-Transformer), [**Autoformer**](https://github.com/thuml/Autoformer).