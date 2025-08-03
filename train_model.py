import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

DATA_DIR = 'data'

X = []  
y = []  

labels = sorted(os.listdir(DATA_DIR))  


for label in labels:
    folder_path = os.path.join(DATA_DIR, label)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            data = np.loadtxt(file_path)
            X.append(data)
            y.append(label)
        except:
            print(f"Error loading {file_path}")

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

with open("sign_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Classes:", sorted(set(y_train)))

print("✅ Model saved as 'sign_model.pkl'")