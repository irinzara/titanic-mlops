import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

def preprocess_data(df):
    # Fill missing Age with median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    # Drop Cabin column
    df = df.drop('Cabin', axis=1)
    # Fill missing Embarked with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    # Encode categorical columns
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return df

def main():
    # Paths
    data_path = os.path.join('data', 'raw', 'train.csv')
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'logistic_regression_model.pkl')

    # Create model directory if not exists
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)

    # Preprocess data
    df = preprocess_data(df)

    # Features and target
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[features]
    y = df['Survived']

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()
