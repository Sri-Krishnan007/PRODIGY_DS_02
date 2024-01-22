import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

titanic_data = pd.read_csv(r"C:\Users\srikr\Downloads\titanic\train.csv")

titanic_data.head()

# Check missing values
titanic_data.isnull().sum()

# Drop duplicates
titanic_data = titanic_data.drop_duplicates()

# Handle missing values
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)


sns.histplot(titanic_data['Age'], kde=True)
plt.title('Distribution of Age')
plt.show()

sns.pairplot(titanic_data[['Survived', 'Pclass', 'Age', 'Fare']])
plt.show()

sns.countplot(x='Survived', hue='Sex', data=titanic_data)
plt.title('Survival Count by Gender')
plt.show()