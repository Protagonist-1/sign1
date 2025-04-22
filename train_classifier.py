import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
try:
    with open('./data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print("Error: data.pickle not found. Run create_dataset.py first.")
    exit()

# Check data consistency and pad shorter samples
max_length = max(len(item) for item in data_dict['data'])
padded_data = [np.pad(item, (0, max_length - len(item)), 'constant') for item in data_dict['data']]
data = np.asarray(padded_data)

# Encode labels numerically
labels = np.asarray(data_dict['labels'])
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)  # Convert text labels (phrases) to numbers

# Print label distribution
unique_labels = set(labels)
print(f"Unique labels: {unique_labels} (Total classes: {len(unique_labels)})")

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate model
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_predict, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}% ✅")

# Save trained model & label encoder
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'label_encoder': label_encoder}, f)

print("✅ Model & Label Encoder saved successfully!")
