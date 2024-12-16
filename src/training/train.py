import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

def preprocess_data(file_path):
    # Load data
    data = pd.read_csv(file_path)

    # Drop Date and Time columns (not needed for modeling)
    data = data.drop(columns=["Date", "Time"], errors='ignore')
    
    # Replace invalid values (-200) with NaN and drop rows with missing values
    data.replace(-200, pd.NA, inplace=True)
    data = data.dropna()

    return data

def train_model():

    # Get the absolute path of the file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Properly resolve the relative path
    data_path = os.path.normpath(os.path.join(current_dir, "../../data/processed/cleaned_train_data.csv"))

    print("data_path", data_path)
    # Preprocess the training data
    train = preprocess_data(data_path)

    # Split into features and target
    X = train.drop("CO(GT)", axis=1)  # All columns except CO(GT)
    y = train["CO(GT)"]  # Target column: CO(GT)

    # Train-Test Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Validation
    predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, predictions)
    print(f"Validation MSE: {mse}")

    # Ensure the 'models' directory exists
    os.makedirs("models", exist_ok=True)

    # Save the trained model
    joblib.dump(model, "models/random_forest.pkl")
    print("Model saved to 'models/random_forest.pkl'.")

    return model

def evaluate_on_test_data(model, test_file_path):
    # Preprocess the test dataset
    test = preprocess_data(test_file_path)

    # Split into features and target
    if "CO(GT)" in test.columns:
        X_test = test.drop("CO(GT)", axis=1)
        y_test = test["CO(GT)"]
        
        # Predict and compute MSE
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Test MSE: {mse}")
    else:
        X_test = test
        predictions = model.predict(X_test)
        print("Predictions generated for test dataset without target column.")

    # Save predictions
    os.makedirs("data/processed", exist_ok=True)  # Ensure directory exists
    test["predictions"] = predictions
    test.to_csv("data/processed/test_predictions.csv", index=False)
    print("Test predictions saved to 'data/processed/test_predictions.csv'.")

if __name__ == "__main__":
    # Train the model
    trained_model = train_model()

    # Get the absolute path of the file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Properly resolve the relative path
    cleaned_train_data_path = os.path.normpath(os.path.join(current_dir, "../../data/processed/cleaned_train_data.csv"))

    print("cleaned_train_data_path", cleaned_train_data_path)
    
    # Evaluate the model on the test dataset
    evaluate_on_test_data(trained_model, cleaned_train_data_path)
