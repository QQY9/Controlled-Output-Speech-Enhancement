A Time-Domain Speech Enhancement Model with Controllable Output Based on Conditional Network
=
The model features diverse output characteristics, allowing control over the network output by adjusting parameter values to accommodate different listeners' needs and preferences.

# Installation
First, install Python 3.7 (recommended with Anaconda).
# Development
git clone https://github.com/QQQQQQQQY/Controlled-Output-Speech-Enhancement.git \
cd denoiser-main1condnet201\
pip install -r requirements_cuda.txt 
# Train and evaluate
## Data
Run `sh make_data.sh` to generate json files. 
## Train
Training is simply done by launching the `train.py` script: This scripts read all the configurations from the conf\config.yaml file.
## Logs
Logs are stored by default in the `outputs` folder. In the experiment folder you will find the `best.th` serialized model, the training checkpoint `checkpoint.th`, and well as the log with the metrics `trainer.log`. All metrics are also extracted to the `history.json` file for easier parsing. 
## Evaluate
1.Set the `laminput` (quantile) values and `samples_dir` (the corresponding storage location for enhanced speech) in the `conf/config.yaml` file. Each `laminput` value represents a specific output characteristic.
For example, `laminput=0.1`; `samples_dir: out1`.\
2.Evaluating the models can be done by:
```
python -m model.evaluate --model_path=<path to the model> --data_dir=<path to folder containing noisy.json and clean.json>
```
Note that the path given to --model_path should be obtained from one of the best.th file, not checkpoint.th.
