# Semafours - MAKEathon 2022 @ FHNW


 - ```train.py``` script to use a ML clustering model using word2Vec
 - @TODO ```predict.py``` script to predict cluster for unseen instance wrapped in Flask API

 
## API Definition for Prediction

### Request Endpoint

```POST http://localhost/predict```

### Request Body (JSON):

```
{
    "title":"Lorem ipsum",
    "abstract":"Pellentesque congue ligula orci, nec iaculis lacus egestas quis. Nullam ut est cursus, porttitor libero eget, faucibus mi."
}
```

### Request Answer (JSON):

The API returns the predicted cluster number (integer) and if possible a score (float) that indicated the confidence in the result. Alternatively, this could be the silhouette coefficient scaled to 0-100% range.

```
{
    "cluster":123,
    "score":0.65,
    "authors": [
        {
            "iri": "http://semopenalex/author/1234",
            "name": "Firstname Lastname",
            "orcid": "1234-1234-5678-0987"
        },
        { 
            "iri": "http://semopenalex/author/5421",
            "name": "Ipsum Lorum",
            "orcid": "8765-6543-6789-5438"
        },
        ...
    ]
}
```

