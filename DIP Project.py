import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --------------------- Logistic Regression Training ---------------------
print("Loading and training Logistic Regression model...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0
y = y.astype(np.int8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logistic_model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=100)
logistic_model.fit(X_train, y_train)
print("Logistic Regression model trained.")

# --------------------- CNN Model Definition ---------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 13 * 13, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --------------------- CNN Training ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_len = int(0.8 * len(train_dataset))
val_len = len(train_dataset) - train_len
train_data, val_data = random_split(train_dataset, [train_len, val_len])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

cnn_model = CNNModel().to(device)
optimizer = optim.Adam(cnn_model.parameters())
criterion = nn.CrossEntropyLoss()

print("Training CNN model...")
for epoch in range(3):
    cnn_model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = cnn_model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete.")
torch.save(cnn_model.state_dict(), "cnn_digit_model.pth")

# --------------------- Utility Functions ---------------------
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    _, thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_images = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= 5 and h >= 10:
            roi = thresh[y:y+h, x:x+w]
            roi = cv2.resize(roi, (28, 28))
            roi_norm = roi.astype('float32') / 255.0
            digit_images.append((x, roi, roi_norm.reshape(1, -1)))
    digit_images_sorted = sorted(digit_images, key=lambda item: item[0])
    return digit_images_sorted

def predict_with_logistic(digits):
    number = ""
    for _, _, roi_flat in digits:
        pred = logistic_model.predict(roi_flat)
        number += str(pred[0])
    return number

def predict_with_cnn(digits):
    model = CNNModel().to(device)
    model.load_state_dict(torch.load("cnn_digit_model.pth", map_location=device))
    model.eval()
    number = ""
    for _, _, roi_flat in digits:
        roi_tensor = torch.tensor(roi_flat.reshape(28, 28)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(roi_tensor)
            _, predicted = torch.max(output.data, 1)
            number += str(predicted.item())
    return number

def classify_number(number):
    length = len(number)
    return {
        9: "Passport",
        11: "Mobile",
        13: "CNIC",
        16: "ATM",
        18: "Check Book"
    }.get(length, "Unknown")

# --------------------- Main Execution ---------------------
def main():
    while True:
        print("\n--- Handwritten Digit Recognition System ---")
        image_path = input("Enter the full path to the digit image: ").strip()
        if not os.path.exists(image_path):
            print("❌ Image file does not exist.")
            continue

        print("\nChoose prediction model:")
        print("1. Logistic Regression")
        print("2. CNN Model")
        model_choice = input("Enter choice (1 or 2): ").strip()

        print("\nChoose expected number type:")
        print("1. Passport (9 digits)")
        print("2. Mobile (11 digits)")
        print("3. CNIC (13 digits)")
        print("4. ATM (16 digits)")
        print("5. Check Book (18 digits)")
        number_type_choice = input("Enter choice (1/2/3/4/5): ").strip()
        expected_type = {
            '1': "Passport",
            '2': "Mobile",
            '3': "CNIC",
            '4': "ATM",
            '5': "Check Book"
        }.get(number_type_choice, "Unknown")

        try:
            digits = preprocess_image(image_path)
            if not digits:
                print("No digits detected.")
                continue

            if model_choice == '1':
                recognized_number = predict_with_logistic(digits)
            elif model_choice == '2':
                recognized_number = predict_with_cnn(digits)
            else:
                print("Invalid model choice.")
                continue

            actual_type = classify_number(recognized_number)
            print("\nRecognized Number:", recognized_number)
            print("Classified As:", actual_type)
            print("Expected Type:", expected_type)

            if actual_type == expected_type:
                print("✅ Match Successful.")
            else:
                print("❌ Mismatch Detected.")

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            plt.imshow(img, cmap='gray')
            plt.title("Input Image")
            plt.axis("off")
            plt.show()

        except Exception as e:
            print(f"Error: {e}")

        cont = input("\nDo you want to enter another image path? (Yes/No): ").strip().lower()
        if cont != "yes":
            print("✅ Exiting program. Thank you!")
            break

if __name__ == "__main__":
    main()
