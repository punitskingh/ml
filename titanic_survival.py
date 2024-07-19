import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn import tree

warnings.filterwarnings('ignore')

df = pd.read_csv("./titanic_train.csv")

print(df.shape)
print(df.columns)
print(df.describe().to_string())

sns.countplot(x="Pclass", data=df, hue='Age')
plt.show()

sns.histplot(df['Age'], kde=False, bins=30)
sns.set_theme(style="ticks")
plt.show()

df['Along'] = df['SibSp'] + df['Parch']
print(df.tail().to_string())

df["Along"] = np.where(df['Along'] > 0, 1, df['Along'])
print(df.tail().to_string())

sns.countplot(x="Sex", data=df, hue='Survived')
plt.show()

df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])
df['Sex'] = np.where(df['Sex'] == 'male', 1, 0)
print(df.head().to_string())

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True)
plt.show()

print(df.isnull().sum())

sns.boxplot(x='Pclass', y='Age', data=df)
plt.show()

print(df[df['Pclass'] == 1]['Age'].mean())
print(df[df['Pclass'] == 2]['Age'].mean())
print(df[df['Pclass'] == 3]['Age'].mean())

def fillage(row):
    age = row[0]
    pclass = row[1]
    if pd.isnull(age):
        if pclass == 1:
            return 38.22
        elif pclass == 2:
            return 29.8
        else:
            return 25.14
    else:
        return age

df['Age'] = df[['Age', 'Pclass']].apply(fillage, axis=1)

print(df.isnull().sum())

X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=0)

print(X_train.shape, y_train.shape)

model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

print(classification_report(y_pred, y_test))
print(f1_score(y_test, y_pred))

f = X.columns
print(f)

plt.figure(figsize=(15, 10))
tree.plot_tree(model, feature_names=f, class_names=['d', 's'])
plt.show()
