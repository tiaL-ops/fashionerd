from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure stopwords are downloaded
nltk.download('stopwords')

# Example dataset with more examples
reviews = [
    "I love these shoes, they are so comfortable!",
    "The quality of this jacket is terrible, very disappointed.",
    "This dress fits perfectly and looks amazing.",
    "I hate the design of this bag, it's so ugly.",
    "These sneakers are stylish and well-made.",
    "The fabric of this shirt feels very cheap, wouldn't recommend it.",
    "I absolutely love this collection, everything is on point!",
    "The pants didn't fit well, returning them.",
    "The handbag looks cheap and poorly made.",
    "This coat is fantastic, so warm and trendy!",
]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1]  # 1: Positive, 0: Negative

# Split the data
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

# Stop words to remove common, irrelevant words
stop_words = set(stopwords.words('english'))

# Create a pipeline that vectorizes the text, applies Logistic Regression, and tunes parameters
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words=stop_words)),  # Vectorize with bigrams and stopwords
    ('clf', LogisticRegression())
])

# Define the hyperparameters to tune
parameters = {
    'tfidf__max_df': [0.7, 1.0],  # Ignore terms with very high frequency (as they might not be informative)
    'tfidf__min_df': [1, 2],      # Minimum document frequency
    'clf__C': [0.1, 1, 10]        # Regularization strength for Logistic Regression
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best parameters from grid search
print("Best parameters from GridSearchCV:")
print(grid_search.best_params_)

# Predict using the best estimator
y_pred = grid_search.predict(X_test)

# Evaluate the model with accuracy, confusion matrix, and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Perform cross-validation for more reliable performance estimate
cross_val_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cross_val_scores}")
print(f"Mean CV score: {cross_val_scores.mean():.2f}")
