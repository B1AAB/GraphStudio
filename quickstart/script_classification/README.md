# _Hello World_ Script Classification

This project provides a simple, end-to-end example of
working with sampled communities from the Bitcoin Graph.


Here, we train and evaluate a model for a script classification task. 
The objective is to cluster a script based on its neighbors 
at a given distance (a 3-hop distance in this example).


The sampling algorithm used to generate these communities 
creates samples by taking a random node and sampling its neighborhood 
using the Forest Fire method. 
Hence, each community has a root node, 
and other nodes are at an n-hop distance from it, 
with directed edges.


We use this property in our model, 
where we train a model using unsupervised contrastive learning to 
generate node embeddings. These embeddings are created by 
aggregating information from a node's neighbors, 
leveraging both the structural and feature properties of its neighborhood. 
Specifically, the model uses two GINEConv layers for message passing 
that incorporates edge features, followed by an APPNP layer 
for feature propagation.
For the root node in a community, 
the different views required for contrastive learning 
are created from slightly jittered node and edge properties of its neighbors
and applies InfoNCE loss during training.


We will then generate node embeddings for the nodes 
in the test dataset and cluster them. 
For each node, we can then check the statistical properties of its neighborhood 
and compare them across other nodes, 
both within the same cluster and in different clusters. 
We will also use labels from WalletExplorer to compare our resulting clusters 
with the wallets identified by WalletExplorer.


Note: _This pipeline is designed to demonstrate how to work with this dataset and is not intended to be a reference implementation._


## Overview 

One effective way to train models on massive graphs 
like the Bitcoin Graph is to first sample smaller subgraphs, aka communities. 
While the sampling process is typically application-specific, 
this demo uses a generic dataset of randomly sampled communities for a quick start. 
These communities are homogeneous, containing only script nodes (addresses) 
and the script-to-script edges. 
Each sampled community is stored in its own directory, 
which contains TSV files for its nodes, edges, and metadata.
For instance:

```bash
.
├── 202509091456271878
│   ├── BitcoinS2S.tsv
│   ├── BitcoinScriptNode.tsv
│   ├── BitcoinScriptNode_Annotated.tsv
│   └── metadata.tsv
└── metadata.tsv
```

## Usage

* Train the model.
    * Download sampled communities.
    * Set `data_root` in `config.json` to the path of the `raw` directory.
    * Set `saves_root` in `config.json`; this is the path where training logs and model status are saved.
    * Open `train.ipynb`
    * Set the virtual environment in the notebook and install dependencies from `requirements.txt`. 
    * Run all the cells in the notebook. 
        * This will result in processing the `raw` data. 
        (see details [in the following](#data-loading)) and training the model described above.
        * Note that the default configurations are set to use only small communities (min 10 and max 100 nodes per graph).
        These values are set so that the training runs in a reasonably short time on limited resources. 
        You may adjust the graph filter or model parameters to experiment with the model.
    * You may view model training logs in tensor board by running: 
        ```shell
        tensorboard --logdir [Log directory]
        ```


* Annotate script nodes
    We annotate script nodes with labels that indicate the wallet they belong to according to WalletExplorer. 
    * Open `../../off_chain_resources/walletexplorer/step_2_annotate_nodes_with_walletexplorer_labels.ipynb`.
    * Set the `raw_sampled_communities_dir` variable to `data_root` configured above.
    * Run all the cells in the notebook.



* Evaluate the model
    * Open `eval.ipynb` and run all the cells.





## Data loading

The `BitcoinScriptsDataset` class 
loads and preprocesses the raw data
by performing the following steps for each community:

1. Reads the raw node and edge files.

2. Filters out communities that don't meet the specified size criteria (e.g., `min_nodes_per_graph`).

3. Constructs node features (`x`) by combining basic graph properties 
(e.g., in/out-degree) with calculated neighborhood statistics (e.g., the sum and mean of incoming transaction values).

4. Creates edge features (`edge_attr`).

5. Converts each community into a `torch_geometric.data.Data` object.


Finally, all processed communities are collated and saved to a single, 
optimized file in the processed directory for efficient loading during training.
