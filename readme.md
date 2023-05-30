# Hyperbolic Diffusion Embedding and Distance for Hierarchical Representation Learning

## Prerequisites

* [Numpy](https://numpy.org/install/)
* [SciPy](https://scipy.org/install/)
* [NetworkX](https://networkx.org)
* [scikit-learn](https://scikit-learn.org/stable/install.html)
* Optional repository for distance evaluation: [MAP & Distortion](https://github.com/HazyResearch/hyperbolics) [1]
* Optional packages for evaluating downstream classification: [Nearest centroid classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html) 
    - random seed: 1234

## Usage Example 

### Graph

```python
from hyperbolic_diffusion_distance import *
from diffusion_operator_util import *

# graph_data is edge connectivity
ev, left_evv, right_evv = diffusion_operator_graph(graph_data, if_full_spec = True)
hde, hdd = hyperbolic_diffusion(ev, left_evv, right_evv, K)
```

### Data Graph

```python
from hyperbolic_diffusion_distance import *
from diffusion_operator_util import *

# data is the high-dimensional observation
# cosine distance used here, other distance can be used to explore in diffusion geometry 
dis_mat = pairwise_distances(data, metric = 'cosine') 
ev, left_evv, right_evv = diffusion_operator_normalized_data(data, dis_mat, if_full_spec = True)
hde, hdd = hyperbolic_diffusion(ev, left_evv, right_evv, K)
```



----------------------------------------------------------------------------
## Experiments 

Download the data below and run hyperbolic diffusion distance experiments using the exp_main.py script

### Hierarchical Graph Embedding Learning

* [Data](https://github.com/HazyResearch/hyperbolics/tree/master/data/edges): the small balanced tree, the phylogenetic tree, the disease, the CS-PHD, and the Gr-Qc graphs
* K: balanced tree (K=3), phylogenetic tree  (K=3), the disease  (K=3), the CS-PHD (K=4), and the Gr-Qc graphs (K=10) 

### Single-Cell Gene Expression Data

* [Data](https://github.com/solevillar/scGeneFit-python/tree/62f88ef0765b3883f592031ca593ec79679a52b4/scGeneFit/data_files) [2]: Zeisel [3] and CBMC [4]
* [Hidden hierarchical structure](https://www.nature.com/articles/s41467-021-21453-4) [2]
* K: Zeisel (K=9), CBMC (K=13)

###  Unsupervised Hierarchical Metric Learning
* [Data](https://archive.ics.uci.edu/ml/datasets.php) [5]: the Zoo, the Iris, the Glass, and the Image Segmentation datasets
* K: Zoo (K=4), Iris(K=6), Glass(K=5), and Image Segmentation (K=8)

----------------------------------------------------------------------------
## Reference 
[1] Sala, F., De Sa, C., Gu, A., and Re, C. Representation
tradeoffs for hyperbolic embeddings. In International
conference on machine learning, pp. 4460–4469. PMLR,
2018.

[2] Dumitrascu, B., Villar, S., Mixon, D. G., and Engelhardt,
B. E. Optimal marker gene selection for cell type discrim-
ination in single cell analyses. Nature communications,
12(1):1–8, 2021.

[3] 
Zeisel, A., Mun ̃oz-Manchado, A. B., Codeluppi, S., Lo ̈nnerberg, P., La Manno, G., Jure ́us, A., Marques, S., Munguba, H., He, L., Betsholtz, C., et al. Cell types in the mouse cortex and hippocampus revealed by single-cell rna-seq. Science, 347(6226):1138–1142, 2015.

[4] Stoeckius, M., Hafemeister, C., Stephenson, W., Houck- Loomis, B., Chattopadhyay, P. K., Swerdlow, H., Satija, R., and Smibert, P. Simultaneous epitope and transcrip- tome measurement in single cells. Nature methods, 14 (9):865–868, 2017.

[5] Dua, D. and Graff, C. UCI machine learning repository, 2017. URL http://archive.ics.uci.edu/ml.

