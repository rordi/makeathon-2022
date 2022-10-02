# adapted from https://dylancastillo.co/nlp-snippets-cluster-documents-using-word2vec/#clean-and-tokenize-data

FILE = './combined_deduplicated.csv'
PICKLE = 'results_model.pkl'
PICKLE_W2V = 'results_w2v_model.pkl'
CLUSTERDATA = 'results_clusters.csv'
SEED = 42
CLUSTERS = 600

import os
import datetime
import time
import random

import nltk
import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from joblib import dump, load

import preprocess

def log(message):
    now = datetime.datetime.now()
    print (now.strftime("%Y-%m-%d %H:%M:%S") + " " + message)

start_time = time.time()

log('Downloading nltk dependencies...')
nltk.download('punkt')
nltk.download("stopwords")

log('Setting random seeds...')
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)



# read csv file
log('Reading CSV file...')
df_raw = pd.read_csv(FILE)

custom_stopwords = set(stopwords.words("english") + ["news", "new", "top"])
text_columns = ["title", "abstract"]

# prepare data frame and create text column based on title + abstract
log('Preparing dataframe with tokenized docs...')
df = df_raw.copy()
for col in text_columns:
    df[col] = df[col].astype(str)

df["text"] = df[text_columns].apply(lambda x: ". ".join(x), axis=1)
df["tokens"] = df["text"].map(lambda x: preprocess.clean_text(x, word_tokenize, custom_stopwords))

docs = df["text"].values
tokenized_docs = df["tokens"].values

log(f"Original dataframe: {df_raw.shape}")
log(f"Pre-processed dataframe: {df.shape}")

log("Pre-processed dataframe head:")
print(df.head())

log('Applying Word2Vec model...')
model = Word2Vec(sentences=tokenized_docs, workers=1, seed=SEED)
model.wv.most_similar("quality")

log('Dumping word2vec model to pickle file... ' + PICKLE_W2V)
dump(model, PICKLE_W2V)



log('Vectorizing documents...')
vectorized_docs = preprocess.vectorize(tokenized_docs, model=model)

def mbkmeans_clusters(
    X,
    k,
    mb,
    print_silhouette_values,
):
    """Generate clusters and print Silhouette metrics using MBKmeans

    Args:
        X: Matrix of features.
        k: Number of clusters.
        mb: Size of mini-batches.
        print_silhouette_values: Print silhouette values per cluster.

    Returns:
        Trained clustering model and labels based on X.
    """
    km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
    log(f"For n_clusters = {k}")
    log(f"Silhouette coefficient: {silhouette_score(X, km.labels_):0.2f}")
    log(f"Inertia:{km.inertia_}")

    if print_silhouette_values:
        sample_silhouette_values = silhouette_samples(X, km.labels_)
        log(f"Silhouette values:")
        silhouette_values = []
        for i in range(k):
            cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
            silhouette_values.append(
                (
                    i,
                    cluster_silhouette_values.shape[0],
                    cluster_silhouette_values.mean(),
                    cluster_silhouette_values.min(),
                    cluster_silhouette_values.max(),
                )
            )
        silhouette_values = sorted(
            silhouette_values, key=lambda tup: tup[2], reverse=True
        )
        for s in silhouette_values:
            print(
                f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
            )
    return km, km.labels_

clustering, cluster_labels = mbkmeans_clusters(
	X=vectorized_docs,
    k=CLUSTERS,
    mb=500,
    print_silhouette_values=True,
)

log('Dumping trained model to pickle file... ' + PICKLE)
dump(clustering, PICKLE)

log("Most representative terms per cluster (based on centroids):")
for i in range(min(CLUSTERS, 50)):
    tokens_per_cluster = ""
    most_representative = model.wv.most_similar(positive=[clustering.cluster_centers_[i]], topn=5)
    for t in most_representative:
        tokens_per_cluster += f"{t[0]} "
    print(f"Cluster {i}: {tokens_per_cluster}")

log('Feeding back cluster numbers to documents data...')
df_clusters = pd.DataFrame({
    "pub": df["pub"],
    # "text": docs,
    # "tokens": [" ".join(text) for text in tokenized_docs],
    "cluster": cluster_labels
})
print(df_clusters.head())

log('Dumping cluster data to CSV file... ' + CLUSTERDATA)
df_clusters.to_csv(CLUSTERDATA)

# script timing
timing = str(time.time() - start_time)
log("Script took " + timing + " seconds to run")
