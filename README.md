# Crop Recommendation System

I built this project to learn how machine learning can be applied to real-world agriculture problems. The idea is simple — given some soil and weather data, the model tells you which crop is best suited to grow.

---

## What it does

Takes 7 inputs:
- N, P, K (nitrogen, phosphorus, potassium levels in soil)
- Temperature
- Humidity
- pH
- Rainfall

And predicts the best crop out of 22 options like rice, maize, coffee, cotton etc.

---

## Dataset

Used a publicly available dataset from Kaggle with 2200 samples.
Link: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

The data was clean — no missing values, no duplicates, balanced classes. Made preprocessing straightforward.

---

## Approach

Nothing too fancy here:
- Encoded the crop labels using LabelEncoder
- Scaled the features using StandardScaler
- Split the data 80/20 for training and testing
- Trained a Decision Tree Classifier

Ended up with **98.63% accuracy** on the test set which I was pretty happy with.

---

## How to run it

Clone the repo and install the dependencies:

```bash
pip install -r requirements.txt
```

Then just run:

```bash
python Model/main.py
```

The dataset downloads automatically via kagglehub so you don't need to do anything extra.

---

## Results

Accuracy: 98.63%
Only 4 samples were misclassified out of 440 in the test set.

---

## Tech used

Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

---

## Notes

This was a beginner level project I did to get comfortable with the sklearn pipeline — data loading, preprocessing, training and evaluation. Planning to try other algorithms on the different projects in the future .
