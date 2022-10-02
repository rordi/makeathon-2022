import pickle
import pandas as pd
import json
from flask import Flask, request, jsonify
import preprocess


from nltk import word_tokenize
from nltk.corpus import stopwords

def make_prediction(data):
    # read the new publication abstract
    # df = pd.read_json('test.json')
    text_columns = ["title", "abstract"]

    df = pd.DataFrame.from_dict(data, orient="index", columns=text_columns)

    custom_stopwords = set(stopwords.words("english") + ["news", "new", "top"])

    for col in text_columns:
        df[col] = df[col].astype(str)

    df["text"] = df[text_columns].apply(lambda x: ". ".join(x), axis=1)
    df["tokens"] = df["text"].map(lambda x: preprocess.clean_text(x, word_tokenize, custom_stopwords))
    tokenized_docs = df["tokens"].values

    w2v_model = pickle.load(open('results_w2v_model.pkl', 'rb'))

    vectorized_docs = preprocess.vectorize(tokenized_docs, model=w2v_model)

    mbkmeans_model = pickle.load(open('results_model.pkl', 'rb'))
    cluster_id = mbkmeans_model.predict(vectorized_docs)

    res = {
        "cluster": cluster_id,
        "score": 0.65
    }

    return res

# API
app = Flask(__name__)

@app.route('/post-json', methods=['POST'])
def process_json():
    # content_type = request.headers.get('Content-Type')
    # if (content_type == 'application/json')
    jsonReq = json.loads(request.data)

    res = make_prediction(jsonReq)

    return jsonify(res)

app.run(debug=True)

# cluster the new publication abstract

# query clusters

# query authors

# sort authors by publications / citentions

# print recommendation authors

