# Insider-Threat-Detection
## Dataset
We use the CMU dataset for experiments, which is a semi-synthetic dataset obtained in cooperation between Carnegie Mellon University and Bigdata, which records various log data of company employees. Specifically, we used the R4.2 version of this dataset, which can be downloaded from the following link:https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099
## Run
Since structured log data cannot be directly used for model training, we need to perform a preprocessing step first. You can use the following commands to perform data preprocessing:</br>
```python utils/preprocess.py```</br>
The logs processed above are arranged in chronological order. For data alignment, different data record granularity can be selected. For example, if converted to hourly accumulation, you can use:</br>
···python utils/op2hour.py```</br>
After completing the above steps, insider threat detection can be performed:</br>
···python train.py```
