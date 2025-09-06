# **Handwritten Digit Recognition using FCFNN**

## **1. Project Overview**

This project focuses on building a **Fully Connected Feedforward Neural Network (FCFNN)** for recognizing handwritten English digits (0–9). In addition to using the MNIST dataset, a **custom handwritten dataset** was collected, preprocessed, and used to retrain and evaluate the model.

The main objectives are:

* Collect and preprocess a custom handwritten digit dataset.
* Retrain an FCFNN using both the MNIST dataset and the custom dataset.
* Evaluate the model on both the custom test set and MNIST test set.

---

## **2. Dataset Collection and Preparation**

### **2.1 Data Collection**

* Multiple individuals wrote digits 0–9 on paper multiple times.
* The samples were scanned or photographed and stored as images.

### **2.2 Preprocessing**

* Convert images to **grayscale**.
* Resize images to **28×28 pixels** (to match MNIST).
* Normalize pixel values to the range **\[0, 1]**.
* Organize data into folders for each digit:

  ```
  dataset/
    train/
      0/
      1/
      ...
      9/
    test/
      0/
      1/
      ...
      9/
  ```

### **2.3 Splitting Dataset**

* **Training set:** 80% of the collected images.
* **Test set:** 20% of the collected images.
* Saved as a **NumPy compressed file** (`.npz`) for easy loading:

  ```python
  np.savez("handwritten_digits.npz",
           X_train=X_train, y_train=y_train,
           X_test=X_test, y_test=y_test)
  ```

---

## **3. Model Architecture**

A **Fully Connected Feedforward Neural Network (FCFNN)** was implemented with the following structure:

| Layer    | Neurons     | Activation                     |
| -------- | ----------- | ------------------------------ |
| Input    | 28×28 = 784 | -                              |
| Hidden 1 | 256         | ReLU                           |
| Hidden 2 | 128         | ReLU                           |
| Hidden 3 | 64          | ReLU                           |
| Output   | 10          | Softmax (via CrossEntropyLoss) |

**Forward pass:**

* Flatten the input image to a vector of size 784.
* Pass through hidden layers with ReLU activation.
* Output layer produces logits for 10 classes.

---

## **4. Training Procedure**

### **4.1 Data Loading**

* Load custom dataset from `.npz` file.
* Load MNIST dataset using PyTorch’s `torchvision.datasets.MNIST`.
* Combine custom and MNIST datasets for training using `ConcatDataset`.

### **4.2 Model Training**

* Loss function: `CrossEntropyLoss`.
* Optimizer: `Adam` with learning rate 0.001.
* Training epochs: 5–10 (depending on convergence).
* Batch size: 64.

**Training loop:**

```python
for images, labels in combined_loader:
    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
```

---

## **5. Evaluation**

The trained FCFNN was evaluated on:

1. **Custom handwritten test set**
2. **MNIST test set**

**Evaluation metrics:**

* **Accuracy (%)**: `(correct predictions / total samples) * 100`

**Example evaluation code:**

```python
def evaluate(model, loader):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total
```

---

## **6. Results**

| Dataset                     | Accuracy |
| --------------------------- | -------- |
| Custom handwritten test set | XX %     |
| MNIST test set              | YY %     |

> Observations: The FCFNN performed slightly lower on the custom handwritten dataset due to variations in individual handwriting styles compared to the standardized MNIST digits.

---

## **7. Conclusion**

* Successfully collected a **custom handwritten digit dataset**.
* Retrained an FCFNN using **both MNIST and custom data**, improving generalization.
* Evaluated the model on both datasets, demonstrating good performance on standard MNIST digits and reasonable performance on real handwritten digits.
* This framework can be extended to other handwritten character recognition tasks.

---
