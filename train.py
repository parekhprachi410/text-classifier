import pandas as pd
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# LOAD DATASET
df = pd.read_csv("bbc_data.csv")

# rename columns
df.columns=['text','label']

# CLEANING
def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z ]',
        '',
        text
    )

    return text

df['text'] = df['text'].apply(clean_text)

# VECTORIZATION
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X = vectorizer.fit_transform(df['text'])

y = df['label']

# SPLIT
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# MODEL
model = LinearSVC()

model.fit(
    X_train,
    y_train
)

pred = model.predict(X_test)

acc = accuracy_score(
    y_test,
    pred
)

print(f"Accuracy: {acc*100:.2f}%")

# SAVE
pickle.dump(
    model,
    open("model.pkl","wb")
)

pickle.dump(
    vectorizer,
    open("vectorizer.pkl","wb")
)