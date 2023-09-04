# DISC
# A pytorch implementation for the IEEE Transactions on Multimedia manuscript "Discrepancy and Structure-based Contrast for Test-time Adaptive Retrieval"

## ENVIRONMENTS
1. pytorch 1.10.0+cu113
2. loguru
3. scikit-learn

## DATASETS
[OFFICE-31]
[OFFICE-HOME]

## Before running the code, the dataset path has to be modified in line 110 and 111 in run.py and line 189 and 195 in officehome.py.

How to train a model:
python run.py --train
It will run the test-time adaptive retrieval task on Art → Real_World from dataset OFFICE-HOME with 64-bit hash codes as default.
To change the task and hash code length, the corresponding arguments have to be modified in run.py.
