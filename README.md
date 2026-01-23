# GraphLinkPrediction
Assignment for my Graph Technologies and Applications course

## Setup

1. **Clone the repository:**
  ```bash
  git clone https://github.com/PanayiotisPerdios/GraphLinkPrediction.git
  cd GraphLinkPrediction
  ```
2. **Make a python3.10 virtual environment**
  ```bash
  python3.10 -m venv venv
  ```
3. **Activate virtual environment**
  ```bash
  source venv/bin/activate
  ```
4. **Update pip and install required packages**
  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
5. **Run Heuristics script**
  ```bash
  python3 heuristics.py
  ```
6. **Run Node2Vec and Simple MLP script**
  ```bash
  python3 node2vec_mlp.py
  ```
7. **Run GNN script**
  ```bash
  python3 gnn.py
  ```   
## (Optional) For Jupyter Notebook use

1. **Install `jupyter` and `ipykernel`**
  ```bash
  pip install jupyter ipykernel
  ```
2. **Register your virtual environment as Jupyter kernel**
  ```bash   
  python -m ipykernel install --user --name venv --display-name "Python 3.10"
  ```
3. **Upgrade Jupyter and related packages**
  ```bash  
  pip install --upgrade \
    notebook \
    jupyter-server \
    jupyterlab \
    traitlets \
    ipykernel
  ```
4. **Launch Jupyter Notebook**
  ```bash
  jupyter notebook
  ```
## Alternatively: VSCode extension for Jupyter:
https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter
