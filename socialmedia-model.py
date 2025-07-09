import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

# 1. Sample dataset
data = {
    "text": [
        "OMG I love this! #fun", 
        "Check out my new photo album!", 
        "Follow me on my journey!", 
        "Just retweeted this cool article", 
        "Had a great time with friends today", 
        "New post on my story!"
    ],
    "label": ["Twitter", "Facebook", "Instagram", "Twitter", "Facebook", "Instagram"]
}
df = pd.DataFrame(data)

# 2. Train pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

pipeline.fit(df['text'], df['label'])

# 3. Save model
joblib.dump(pipeline, 'socialmedia_model.pkl')

# 4. Convert to ONNX
initial_type = [('input', StringTensorType([None, 1]))]
onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
with open("socialmedia_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
