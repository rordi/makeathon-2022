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
    "abstract":"Pellentesque congue ligula orci, nec iaculis lacus egestas quis. Nullam ut est cursus, porttitor libero eget, faucibus mi. Etiam et blandit felis, at convallis tellus. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse potenti. Nunc bibendum sem eget eros condimentum auctor. Aliquam sodales commodo turpis malesuada pretium. Donec sagittis ex id congue cursus. Etiam tempus lacus a augue tincidunt consequat. Nunc vestibulum condimentum sapien, et dapibus purus dapibus nec. Nam sagittis, sapien quis iaculis sagittis, arcu diam semper lectus, nec fermentum libero ligula vel tellus. Cras scelerisque tempus enim vel sodales."
}
```

### Request Answer (JSON):

The API returns the predicted cluster number (integer) and if possible a score (float) that indicated the confidence in the result. Alternatively, this could be the silhouette coefficient scaled to 0-100% range.

```
{
    "cluster":123
    "score":0.65
}
```

