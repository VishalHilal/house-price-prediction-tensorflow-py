import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ----------------------------
# 1. Load Dataset from CSV
# ----------------------------

# Load training data
train_data = pd.read_csv('house_train_data.csv')

# Select relevant features for house price prediction
features = ['GrLivArea', 'BedroomAbvGr', 'YearBuilt']
target = 'SalePrice'

# Extract features and target
X = train_data[features].values
y = train_data[target].values

# Handle missing values
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

print(f"Dataset shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Sample features: {X[0]}")
print(f"Sample price: ${y[0]:,.0f}")

# ----------------------------
# 2. Train / Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 3. Feature Scaling
# ----------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# 4. Build Neural Network
# ----------------------------

model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1)
])

# ----------------------------
# 5. Compile Model
# ----------------------------

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
)

# ----------------------------
# 6. Train Model
# ----------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=200,
    validation_split=0.2,
    verbose=1
)

# ----------------------------
# 7. Evaluate Model
# ----------------------------

loss, mae = model.evaluate(X_test, y_test)

print("\nTest MAE:", mae)

# ----------------------------
# 8. Load Test Data and Make Predictions
# ----------------------------

# Load test data
test_data = pd.read_csv('house_test_data.csv')

# Extract features from test data
X_test_final = test_data[features].values
X_test_final = imputer.transform(X_test_final)
X_test_final_scaled = scaler.transform(X_test_final)

# Make predictions on test data
test_predictions = model.predict(X_test_final_scaled)

# Display some predictions
print("\nTest Data Predictions:")
print(f"Number of test samples: {len(test_predictions)}")
print(f"Sample predictions: ${test_predictions[:5].flatten()}")

# ----------------------------
# 9. Predict New House Price
# ----------------------------

size = float(input("Enter house size (sqft): "))
bedrooms = int(input("Enter number of bedrooms: "))
year_built = int(input("Enter year built: "))

new_house = np.array([[size, bedrooms, year_built]])
new_house_imputed = imputer.transform(new_house)
new_house_scaled = scaler.transform(new_house_imputed)

prediction = model.predict(new_house_scaled)

print(f"\nEstimated price: ${prediction[0][0]:,.0f}")

# ----------------------------
# 10. Save Model
# ----------------------------

model.save("house_price_model.keras")

print("\nModel saved successfully.")
