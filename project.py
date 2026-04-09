import pandas as pd

df = pd.read_csv("data.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df['churn'].value_counts())
print(df['churn'].value_counts(normalize=True) * 100)
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='churn', data=df)
plt.show()
sns.boxplot(x='churn', y='monthly_charges', data=df)
plt.show()
sns.histplot(data=df, x='tenure', hue='churn')
plt.show()
print(df.groupby('churn')['monthly_charges'].mean())
df['gender'] = df['gender'].map({'Male':1, 'Female':0})
df['churn'] = df['churn'].map({'Yes':1, 'No':0})

print(df.head())
X = df.drop('churn', axis=1)
y = df['churn']

print(X.head())
print(y.head())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

print("Model trained ✅")
y_pred = model.predict(X_test)

print("Predictions:", y_pred)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)