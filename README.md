# CXR-ViewPosition-SSL
Comparing semi-supervised learning algorithms for CXR view position classification
Data: Stanford dataset (To fasten training, I only used 10,000 images for training)

### Supervised learning
```
python3 supervised-train.py --gpu=0 --out='result/supervised' 
```

### Semi-supervised learning
#### MixMatch
```
python3 ssl-train.py --gpu=1 --out='result/mixmatch' --model='mixmatch' --val-iteration=1024 --n-labeled=1000 
```
