Exploring  Dynamic  Context  for  Multi-path  Trajectory  Prediction
===


DCENet Structure
===
![DCENet](https://github.com/tanjatang/DCENet/blob/master/pipeline/pipeline.png)


#### Requiements
* python3
* keras-gpu 2.3.1
* tensorflow 2.1.0
* numpy
...

```
pip install -r requiements.txt
```
 
#### Data Preparation
1. download raw data from directory /WORLD H-H TRAJ

2. run /scripts/trainer.py by setting arg.preprocess_data==True for data processing.
**Note:** check if all of the listed directories have already been yieled; set arg.preprocess_data==False if you have already the processed data.

#### Test
You can get the results as reported in the paper using our pretrained model.
1. Download pretrained model from /models/best.hdf5

#### Train
You also can train from sratch by /scripts/trainer.py


#### Citation

If you find our work useful for you, please cite it as:
----
```html
@article{cheng2020exploring,
  title={Exploring Dynamic Context for Multi-path Trajectory Prediction},
  author={Cheng, Hao and Liao, Wentong and Tang, Xuejiao and Yang, Michael Ying and Sester, Monika and Rosenhahn, Bodo},
  journal={arXiv preprint arXiV:2010.16267},
  year={2020}
}
```
