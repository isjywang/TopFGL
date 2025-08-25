## TopFGL: A Topology-Aware and Distribution-Agnostic Federated Learning Framework Tackling Topological Heterogeneity on Graph Data

**Requirements**
1. The required dependencies are listed in `./TopFGL/requirements.txt`. You can install them using pip:
```python
pip install -r requirements.txt
```
2. For graph data partitioning, we use the Metis library. Please install it by following the instructions at this repository: https://github.com/james77777778/metis_python.


**Data Preparation and Partitioning**
1. __Create a Dataset Directory:__ Navigate to the `/TopFGL/datasets/` directory and create a new folder named after the dataset you intend to use (e.g., Cora and 10 clients).
```python
cd ./TopFGL/datasets/
mkdir ./Cora
cd ./Cora
mkdir ./10
```
2. __Partition the graph:__ Open the file `/TopFGL/datasets/get_data_heterogeneity.py` and fill in the required information to configure the data partitioning process.
Then please run the data preprocess code:
```python3 
cd /TopFGL/datasets/
python get_data_heterogeneity.py
```

3. __Initialize Topological Embeddings:__  This step generates the initial topological embeddings for the dataset. Please open `/TopFGL/lib/init_topological.py` and specify your dataset's information. Once configured, execute the code to start the initialization:
```python3 
cd /TopFGL/lib/
python init_topological.py
```

**Training**

1. Run Local Training:
```python3
python main.py --alg local --dataset Cora --clients 10
```
2. Run FedAvg:
```python3
python main.py --alg fedavg --dataset Cora --clients 10
```
3. Run TopFGL:
```python3
python main.py --alg topfgl -dataset Cora --clients 10
```