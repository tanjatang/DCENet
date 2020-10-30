Exploring  Dynamic  Context  for  Multi-path  Trajectory  Prediction
===


DCENet
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
Create necessary directories
```
python /scripts/mkdir.py
```
 
#### Data Preparation
1. download raw data from xxxx, and save in xxxxx
2. run /scripts/trainer.py by setting arg.preprocess=True for data processing.
**Note:** set arg.preprocess=False when you have already the processed data when you run /scripts/trainer.py to save time.

#### Test
You can get the results as reported in the paper using our pretrained model.
1. Download pretrained model (give a link)
2. 

#### Train
You also can train from sratch by /scripts/trainer.py



#### Bibtex

If you find our work useful for you, please cite it as:
----
```html
@article{cheng2020exploring,
  title={Exploring Dynamic Context for Multi-path Trajectory Prediction},
  author={Cheng, Hao and Liao, Wentong and Tang, Xuejiao and Yang, Michael Ying and Sester, Monika and Rosenhahn, Bodo},
  journal={arXiv preprint},
  year={2020}
}
```
