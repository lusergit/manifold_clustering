# Clustering of manifold
The following project is based on the model proposed by
[n2d](https://arxiv.org/abs/1908.05968). We'll use the pytorch library
in order to build an autoencoder and train it, take the data encoded
in its encoded space and shape it on a manifold to better cluster it
with a shallow cluster method, in order to boost the performance of
the shallow clustering.

The corresponding colab notebook can be found
[here](https://colab.research.google.com/drive/11j16mhZbaPT_2RA7rb6F1c5JRcn2RMYe?usp=sharing]

## Project init
To init the project you can either create a conda environment
(recomended, since this way the python version is controlled) with
```sh
conda create --name torch python=3.8
conda activate torch
pip install -r requirements.txt
```

for better performances is also recomended to install
[MulticoreTSNE](https://github.com/DmitryUlyanov/Multicore-TSNE)
trough

```sh
pip install git+https://github.com/jorvis/Multicore-TSNE
```

but the project should also work (just a bit slowly) with the TSNE
manifold include in the `sklearn` library (included in the
`requirements.txt`)

## Run the script
Once the libraries are loaded to run the script one can just
```sh
python manifold_clustering.py [OPTIONS]
```
where options can be
- **--load PATH** to load the weights of the autoencoder on the
  desired path
- **--save PATH** to save the weights of the next training session on
  the specified path
- **--no_shallow** to *not* run the shallow algorithms to produce the
  baseline
- **--no_umap** to *not* run the umap manifold clustering
- **--no_tsne** to *not* run the t-SNE manifold clustering
- **--no_isomap** to *not* run the isomap manifold clustering

The script will run, printing what it's doing at each step
