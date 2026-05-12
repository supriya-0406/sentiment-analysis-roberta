import pandas as pd
from transformers import pipeline

# Load dataset
df = pd.read_csv("reviews.csv")

print("Dataset Preview:")
print(df.head())

# Load sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# Predict sentiment
results = []

for review in df['Review']:
    result = classifier(review)[0]
    results.append(result['label'])

# Add predictions
df['Predicted_Sentiment'] = results

print("\nPrediction Results:")
print(df)
