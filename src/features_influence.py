import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Load the dataset
file_path = '/home/wytcor/PROJECTs/mestrado-ufes/lab-life/EDA_pad_ufes_20/data/merged_metadata.csv'
dataset = pd.read_csv(file_path)

# Define target and features
target_column = "diagnostic"
feature_columns = [col for col in dataset.columns if col not in ["patient_id", "lesion_id", "img_id", target_column]]

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


# # Nome das features
# encoded_feature_names = features_encoded.columns

# # Retrieve model parameters
# log_prob_features = nb_model.feature_log_prob_  # Log probabilities of features given a class
# class_log_priors = nb_model.class_log_prior_  # Log priors for each class

# # Choose a single sample
# sample_idx = 0
# sample = X_test[sample_idx]

# # Compute the contribution of each feature for each class
# feature_contributions = []
# for class_idx in range(log_prob_features.shape[0]):
#     class_contributions = (
#         sample * log_prob_features[class_idx] +
#         (1 - sample) * np.log(1 - np.exp(log_prob_features[class_idx]))
#     )
#     feature_contributions.append(class_contributions)

# # Convert to a DataFrame for analysis
# contributions_df = pd.DataFrame(
#     feature_contributions,
#     columns=encoded_feature_names,
#     index=[f"Class_{i}" for i in range(log_prob_features.shape[0])]
# )

# # Sum feature contributions and add the log priors
# total_contributions = contributions_df.sum(axis=1) + class_log_priors

# # Normalize contributions to probabilities
# probabilities = np.exp(total_contributions - np.max(total_contributions))
# probabilities /= probabilities.sum()

# # Rank features by their contributions for the predicted class
# predicted_class_idx = probabilities.argmax()
# predicted_class_contributions = contributions_df.iloc[predicted_class_idx]
# ranked_contributions = predicted_class_contributions.sort_values(ascending=False)

# # Print the top features contributing to the predicted class
# print("Feature Contributions for Predicted Class:")
# print(ranked_contributions.head(10))

# # Optionally, visualize contributions
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
# ranked_contributions.head(10).plot(kind='bar', color='skyblue')
# plt.title(f"Top 10 Feature Contributions for Predicted Class {predicted_class_idx}")
# plt.ylabel("Log Probability Contribution")
# plt.xlabel("Feature")
# plt.show()
