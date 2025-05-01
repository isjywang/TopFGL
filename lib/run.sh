## unzip the data in ../datasets/
## init the data in init_structural.py

# run local
python3 main.py --alg local  --dataset Cora

# run fedavg
python3 main.py --alg fedavg --dataset Cora

# run fedtop
python3 main.py --alg fedtop --dataset Cora