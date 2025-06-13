## 1. unzip the data in ../datasets/
## 2. init the data in init_structural.py

# run local
python3 main.py --alg local  --dataset Cora

# run fedavg
python3 main.py --alg fedavg --dataset Cora

# run topfgl
python3 main.py --alg topfgl --dataset Cora