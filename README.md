# GraphSearchNet
This repo is the implementation of the paper "GraphSearchNet: Enhancing GNNs via Capturing Global Dependency for Semantic Code Search". We encode the programs and descriptions in a dual graph framework
to learn the semantic similarity.
## Get started
### Prerequisites
This code is written in python 3. You will need to install the required packages in order to run the code.
We recommend to use conda virtual environment for the package management. Install the package requirements with ```pip install -r requirements.txt```.
### Preprocess
1. Download the raw data from [CodeSearchNet](https://github.com/github/CodeSearchNet). 
2. For Java Graph Construction, please run the scripts at graph-based-search/parsers/sourcecode/java/build_java_graph.py
3. For Python Graph Construction, please run the scripts at graph-based-search/parsers/sourcecode/java/build_python_graph.py
We provide the partial data in this repo to help run our model quickly. The whole processed data will be public.
### Run the model
1. Train Java model ```python main.py --config config/search_train_java.yml```
2. Train Python model ```python main.py --config config/search_train_puython.yml```

We also provide the queried results generated by our model based on the 99 queries from CodeSearchNet in graph-based-search/answers.

## Citation
If you find this code or our paper relevant to your work, please cite our arXiv paper:

```
@article{liu2021graphsearchnet,
  title={GraphSearchNet: Enhancing GNNs via Capturing Global Dependency for Semantic Code Search},
  author={Liu, Shangqing and Xie, Xiaofei and Ma, Lei and Siow, Jingkai and Liu, Yang},
  journal={arXiv preprint arXiv:2111.02671},
  year={2021}
}
```