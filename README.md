# smart-home-optimizer

Creating a comprehensive Python program to optimize energy usage in smart homes using machine learning involves several steps. We'll develop a basic structure that handles data loading, preprocessing, model training, and predictions while ensuring error handling and good coding practices. For simplicity, we'll use a basic machine learning model like Random Forest and some synthetic data.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SmartHomeOptimizer:
    def __init__(self, data_file):
        self.data_file = data_file
        self.model = RandomForestRegressor(random_state=42)
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_data(self):
        """Loads data from a CSV file."""
        try:
            self.data = pd.read_csv(self.data_file)
            logging.info(f"Data loaded successfully from {self.data_file}")
        except FileNotFoundError:
            logging.error(f"File not found: {self.data_file}")
            raise
        except pd.errors.ParserError:
            logging.error(f"Error parsing the file: {self.data_file}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise

    def preprocess_data(self):
        """Preprocess the data, including handling missing values and splitting."""
        try:
            # Example preprocessing; actual steps depend on the dataset
            self.data.fillna(self.data.mean(), inplace=True)
            logging.info("Missing values handled.")

            # Assuming 'energy_usage' is the target variable
            X = self.data.drop('energy_usage', axis=1)
            y = self.data['energy_usage']

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logging.info("Data pre-processing and splitting completed.")
        except KeyError as e:
            logging.error(f"Column not found in dataset: {e}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred during preprocessing: {e}")
            raise

    def train_model(self):
        """Trains the machine learning model."""
        try:
            self.model.fit(self.X_train, self.y_train)
            logging.info("Model training completed.")
        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")
            raise

    def evaluate_model(self):
        """Evaluates the model and prints the Mean Squared Error."""
        try:
            predictions = self.model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, predictions)
            logging.info(f"Model evaluation completed with MSE: {mse:.2f}")
        except Exception as e:
            logging.error(f"An error occurred during model evaluation: {e}")
            raise

    def optimize_energy(self, new_data):
        """Predicts optimal energy usage for new data."""
        try:
            if isinstance(new_data, pd.DataFrame):
                predictions = self.model.predict(new_data)
                logging.info(f"Energy optimization completed with predictions: {predictions}")
                return predictions
            else:
                raise ValueError("Input data should be a pandas DataFrame")
        except ValueError as e:
            logging.error(f"Value error: {e}")
            raise
        except Exception as e:
            logging.error(f"An unknown error occurred while optimizing energy: {e}")
            raise

if __name__ == "__main__":
    # Initialize the optimizer with the path to your data file
    optimizer = SmartHomeOptimizer('smart_home_data.csv')
    
    # Load and preprocess data
    optimizer.load_data()
    optimizer.preprocess_data()
    
    # Train and evaluate model
    optimizer.train_model()
    optimizer.evaluate_model()
    
    # Example usage with new input data for prediction
    sample_new_data = pd.DataFrame({
        'feature1': [0.5],
        'feature2': [1.5],
        'feature3': [2.5],
        # Add more features here as per the dataset
    })
    
    try:
        predictions = optimizer.optimize_energy(sample_new_data)
        print("Optimized energy predictions:", predictions)
    except Exception as e:
        logging.error("An error occurred during energy optimization.")
```

**Explanation:**
- **Data Handling:** We load data from a CSV file and handle potential file errors.
- **Preprocessing:** Includes handling missing values and splitting the dataset into training and testing sets.
- **Modeling:** A RandomForestRegressor is trained on the data.
- **Error Handling:** Errors are logged, and exceptions are raised to ensure clean error propagation.
- **Prediction:** Provides a method for predicting energy usage for new data points.

Remember to install the required Python libraries, such as `pandas`, `numpy`, and `scikit-learn`, before running this program. The real-world application requires real data and potentially a more sophisticated model for better performance.