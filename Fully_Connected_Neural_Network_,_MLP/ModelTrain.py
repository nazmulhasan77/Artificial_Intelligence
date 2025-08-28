from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
import numpy as np

# Polynomial function
def my_polynomial(x):
    return 7*x**4 + 5*x**3 + 2*x**2 - 7*x + 10

# Generate dataset
def data_process():
    n = 1000
    x = np.random.randint(0, 100, n)  # Random integers 0-99
    y = np.array([my_polynomial(xi) for xi in x])
    x = x.reshape(-1, 1)  # Make it 2D for Keras
    y = y.reshape(-1, 1)
    return x, y

# Split dataset
def prepare_train_test_val():
    x, y = data_process()
    total_n = len(x)
    train_n = int(total_n * 0.7)
    val_n = int(total_n * 0.1)

    trainX = x[:train_n]
    trainY = y[:train_n]

    valX = x[train_n:train_n+val_n]
    valY = y[train_n:train_n+val_n]

    testX = x[train_n+val_n:]
    testY = y[train_n+val_n:]

    return (trainX, trainY), (valX, valY), (testX, testY)

# Build the model
def build_model():
    inputs = Input((1,))
    h1 = Dense(8, activation='relu', name='Hidden_Layer_1')(inputs)
    h2 = Dense(16, activation='relu', name='Hidden_Layer_2')(h1)
    h3 = Dense(4, activation='relu', name='Hidden_Layer_3')(h2)
    outputs = Dense(1, name='Output_Layer')(h3)
    
    model = Model(inputs, outputs)
    model.summary(show_trainable=True)
    return model

# Main function
def main():
    model = build_model()
    model.compile(optimizer='adam', loss='mse')

    (trainX, trainY), (valX, valY), (testX, testY) = prepare_train_test_val()

    # Train the model
    model.fit(trainX, trainY, validation_data=(valX, valY), epochs=50, batch_size=32)

    # Evaluate on test data
    test_loss = model.evaluate(testX, testY)
    print(f"\nTest Mean Squared Error: {test_loss:.4f}")

    # Predict first 5 samples
    y_pred = model.predict(testX[:5])
    print("\nSample predictions vs true values:")
    for i in range(5):
        print(f"Predicted: {y_pred[i][0]:.2f}, True: {testY[i][0]}")

if __name__ == '__main__':
    main()
