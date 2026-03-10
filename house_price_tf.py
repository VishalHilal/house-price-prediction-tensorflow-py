import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1. Dataset
# ----------------------------

# features: [size_sqft, bedrooms, age]
X = np.array([
    [1200, 3, 20],
    [1500, 4, 15],
    [800, 2, 30],
    [2000, 4, 8],
    [1700, 3, 12],
    [2200, 5, 5],
    [900, 2, 25],
    [1400, 3, 18],
    [1600, 3, 10],
    [2400, 5, 3],
    [3000, 4, 2],
    [1100, 3, 20],
    [1000, 2, 28],
    [1900, 4, 7],
    [2100, 4, 6],
    [2500, 5, 4],
], dtype=float)

# prices (in thousands)
y = np.array([
    200, 250, 120, 340, 300, 400, 150, 230,
    310, 450, 520, 210, 170, 360, 390, 470
], dtype=float)

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
# 8. Predict New House Price
# ----------------------------

size = float(input("Enter house size (sqft): "))
bedrooms = int(input("Enter number of bedrooms: "))
age = int(input("Enter house age: "))

new_house = np.array([[size, bedrooms, age]])
new_house_scaled = scaler.transform(new_house)

prediction = model.predict(new_house_scaled)

print("\nEstimated price: $", prediction[0][0] * 1000)

# ----------------------------
# 9. Save Model
# ----------------------------

model.save("house_price_model.keras")

print("\nModel saved successfully.")
