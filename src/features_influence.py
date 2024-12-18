import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def main(dataset, features_to_be_removed, target):
    # Dropar dados onde "train" é true
    dataset = dataset[dataset["train"] != True]
    
    feature_columns = [col for col in dataset.columns if col not in features_to_be_removed]

    # Identify categorical and numerical columns
    categorical_columns = dataset[feature_columns].select_dtypes(include=["object"]).columns
    numerical_columns = dataset[feature_columns].select_dtypes(include=["number"]).columns
    

    # Apply one-hot encoding to categorical columns
    features_encoded = pd.get_dummies(dataset[feature_columns], columns=categorical_columns, drop_first=True)

    # Handle missing values for all features
    imputer = SimpleImputer(strategy="median")
    features_imputed = imputer.fit_transform(features_encoded)

    # Encode the target variable
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(dataset[target_column])

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_imputed)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42, shuffle=True)


    # Train the Naive Bayes classifiers
    nb_model = BernoulliNB()
    nb_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = nb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_report_result = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

    # Print results
    print("Model Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_report_result)

if __name__=="__main__":
    # Load the dataset
    file_path = '/home/wyctor/PROJETOS/pad_ufes_20_eda/src/results/inference-results/merged_metadata.csv'
    dataset = pd.read_csv(file_path)
    # Features a serem desconsideradas
    # Definir qual a feature a ser predita e quais não serão usadas
    target_column = "diagnostic"
    features_to_be_removed=["patient_id", "lesion_id", "img_id" , "modelo_name", "diagnostic_ACK" , "diagnostic_BCC", "diagnostic_MEL", "diagnostic_NEV", "diagnostic_SCC", "diagnostic_SEK", target_column]
    main(dataset, features_to_be_removed, target_column)