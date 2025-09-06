from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
def load_data():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    return (trainX, trainY), (testX, testY)

# Build model
def build_model():
    inputs = Input((28,28))
    x1 = Flatten()(inputs)
    x2 = Dense(128, activation='relu')(x1)
    x3 = Dense(256, activation='relu')(x2)
    x4 = Dense(512, activation='relu')(x3)
    x5 = Dense(256, activation='relu')(x4)
    x6 = Dense(128, activation='relu')(x5)
    outputs = Dense(10, activation='softmax')(x6)
    model = Model(inputs, outputs)
    return model

# Main function
def main():
    (trainX, trainY), (testX, testY) = load_data()

    model = build_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    history = model.fit(trainX, trainY, validation_split=0.1, epochs=5, batch_size=32, verbose=1)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(testX, testY, verbose=0)
    print(f"\n✅ Test Accuracy: {test_acc:.4f}")

    # Predict on test set
    predictions = model.predict(testX)
    predicted_labels = np.argmax(predictions, axis=1)

    # Plot training history
    plt.figure(figsize=(8,5))
    plt.plot(history.history['accuracy'], label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], label="Val Accuracy")
    plt.xlabel("No of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.savefig("training_accuracy.png")
    plt.legend()
    plt.show()

    # Show 20 test images with predicted labels
    plt.figure(figsize=(12, 8))
    for i in range(20):
        plt.subplot(5, 5, i+1)
        plt.imshow(testX[i], cmap='gray')
        plt.title(f"Pred: {predicted_labels[i]}\nTrue: {testY[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("test_predictions.png")
    plt.show()

if __name__ == "__main__":
    main()
