
# Sentiment Analysis on Women's Clothing E-Commerce Reviews
​
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
​
# Load dataset
df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
df = df[['Review Text', 'Rating']].dropna()
​
# Add Sentiment column: 1 for Positive (Rating >= 4), 0 for Negative
df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)
​
# Visualize Sentiment distribution
sns.countplot(data=df, x='Sentiment')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment (0 = Negative, 1 = Positive)")
plt.ylabel("Count")
plt.show()
​
# TF-IDF Vectorization
X = df['Review Text']
y = df['Sentiment']
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)
​
# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)
​
# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
​
# Predictions and Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
​
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
