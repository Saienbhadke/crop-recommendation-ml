import kagglehub 
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,  plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ----------- Download Dataset --------------

path = kagglehub.dataset_download("atharvaingle/crop-recommendation-dataset")
print("Path to dataset files:", path)

# ----------- Load into Dataframe -----------

Cropset = os.path.join(path, "Crop_recommendation.csv")
df = pd.read_csv(Cropset)
print(df.head())

# ----------- Dataprocessing ---------------- 

print(df.shape)
print(df.isnull())
print(df.isnull().sum())
print(df.duplicated().sum())

x = df.drop("label", axis = 1)
y = df["label"]


le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)

# ------------ Spliting the  Data --------------

x_train , x_test , y_train , y_test = train_test_split(x_scaler ,y_encoded, test_size=0.20 , random_state=42) 


# ------------ Train Model using DecisionTree --------------- 

dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train , y_train)


# ------------- Test Model using DecisionTree ---------------

y_pred = dt.predict(x_test)

print("Accuracy :", accuracy_score(y_test , y_pred))

print("classification report :", classification_report(y_test , y_pred))

print("confusion matrix :", confusion_matrix(y_test , y_pred))


# ------------- Visulization of tree diagram using Matplotlib --------------

plt.figure(figsize=(20, 10))
plot_tree(
    dt,
    feature_names=x.columns,
    class_names=le.classes_,
    filled=True,
    rounded=True,
    max_depth=3
)
plt.title("Decision Tree - Crop Recommendation")
plt.show()


# ---------------- Visulization of Data using seaborn Heatmap -----------------

plt.figure(figsize=(16, 12))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

