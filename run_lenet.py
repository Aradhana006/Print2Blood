"""Train and test the LeNet blood-group model from code/Lenet/lenet.ipynb."""
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT, "dataset")
os.chdir(DATASET_DIR)

file_path = "dataset_blood_group"
name_class = os.listdir(file_path)
print("classes:", name_class)

filepaths = list(glob.glob(file_path + "/**/*.*"))
print("image count:", len(filepaths))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepath = pd.Series(filepaths, name="Filepath").astype(str)
Labels = pd.Series(labels, name="Label")
data = pd.concat([filepath, Labels], axis=1)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
print(data.head(5))

counts = data.Label.value_counts()
sns.barplot(x=counts.index, y=counts.values)
plt.xlabel("Type")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "graphs", "lenet_class_distribution.png"))
plt.close()

train, test = train_test_split(data, test_size=0.20, random_state=42)
print("train:", len(train), "test:", len(test))

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train,
    x_col="Filepath",
    y_col="Label",
    target_size=(224, 224),
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=42,
)
valid_gen = test_datagen.flow_from_dataframe(
    dataframe=test,
    x_col="Filepath",
    y_col="Label",
    target_size=(224, 224),
    class_mode="categorical",
    batch_size=32,
    shuffle=False,
    seed=42,
)

model = Sequential([
    Conv2D(6, kernel_size=(5, 5), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(16, kernel_size=(5, 5), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(120, activation="relu"),
    Dense(84, activation="relu"),
    Dense(8, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(train_gen, validation_data=valid_gen, epochs=20)

pd.DataFrame(history.history)[["accuracy", "val_accuracy"]].plot()
plt.title("Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "graphs", "lenet_accuracy.png"))
plt.close()

pd.DataFrame(history.history)[["loss", "val_loss"]].plot()
plt.title("Loss")
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "graphs", "lenet_loss.png"))
plt.close()

results = model.evaluate(valid_gen, verbose=0)
print(f"Test Loss: {results[0]:.5f}")
print(f"Test Accuracy: {results[1] * 100:.2f}%")

pred = model.predict(valid_gen)
pred = np.argmax(pred, axis=1)
class_labels = dict((v, k) for k, v in train_gen.class_indices.items())
pred = [class_labels[k] for k in pred]
y_test = list(test.Label)
print(classification_report(y_test, pred))

model_path = os.path.join(ROOT, "code", "Lenet", "model_blood_group_detection_lenet.keras")
model.save(model_path)
print("saved:", model_path)

loaded = load_model(model_path)
labels = {"A+": 0, "A-": 1, "AB+": 2, "AB-": 3, "B+": 4, "B-": 5, "O+": 6, "O-": 7}
labels = dict((v, k) for k, v in labels.items())
img_path = os.path.join(file_path, "A+", "cluster_0_1001.BMP")
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
result = loaded.predict(x)
predicted_class = int(np.argmax(result))
predicted_label = labels[predicted_class]
confidence = float(result[0][predicted_class] * 100)
print(f"sample {img_path}")
print(f"Prediction: {predicted_label} with confidence {confidence:.2f}%")
