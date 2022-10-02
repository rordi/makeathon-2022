import pandas as pd
import json
from flask import Flask, request, jsonify
import train

# API
app = Flask(__name__)

@app.route('/post-json', methods=['POST'])
def process-json():
    #content_type = request.headers.get('Content-Type')
    #if (content_type == 'application/json')
    record = json.loads(request.data)
    return jsonify(record)


# read the new publication abstract
df = pd.read_json('test.json')
print(df.to_string())
clean-text(df)
df-raw

# run it through the model
pickleFile = "results_model.pki"
loaded_model = pickle.load(open(pickleFile, 'rb'))
result = loaded_model.predict()

# cluster the new publication abstract

# query clusters

# query authors

# sort authors by publications / citentions

# print recommendation authors

