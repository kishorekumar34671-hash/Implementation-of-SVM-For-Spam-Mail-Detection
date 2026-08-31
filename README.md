# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the email dataset containing email messages and their labels (Spam or Not Spam).

2.Preprocess the email text by converting it into numerical features using TF-IDF vectorization.

3.Train the SVM classifier using the extracted features and the corresponding email labels.

4.Test the model with new emails and classify them as Spam or Not Spam, then evaluate the model accuracy. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: KISHORE KUMAR B
RegisterNumber:212225240073
# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Display first 5 rows
print(data.head())

# Select message and label columns
# For the common spam.csv dataset:
X = data['v2']       # Email/SMS message
y = data['v1']       # ham or spam

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english'
)

X = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create SVM model
model = SVC(kernel='linear')

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test with a new message
new_message = ["Congratulations! You have won a free prize."]

# Convert the new message into TF-IDF features
new_message_vector = vectorizer.transform(new_message)

# Predict
prediction = model.predict(new_message_vector)

print("\nNew Message:")
print(new_message[0])

print("Prediction:", prediction[0])
  
*/
```

## Output:
<img width="1920" height="1080" alt="Screenshot 2026-08-31 204802" src="https://github.com/user-attachments/assets/7b89194a-3a03-470b-94e7-aefeae00325c" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
