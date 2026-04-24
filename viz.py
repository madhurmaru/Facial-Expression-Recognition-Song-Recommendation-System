# fer_cnn_xai_inference.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from keras.models import load_model
from keras.utils import to_categorical

# XAI imports
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries

# --------------------- Load Dataset ---------------------
csv_path = 'fer2013.csv'
df = pd.read_csv(csv_path)

# Filter out rows with incorrect number of pixel values
df['pixel_count'] = df['pixels'].apply(lambda x: len(x.split()))
df = df[df['pixel_count'] == 48 * 48].drop('pixel_count', axis=1)

# Convert pixel strings to 48x48 numpy arrays
X = np.array([np.fromstring(pixels, sep=' ').reshape(48, 48, 1) for pixels in df['pixels']])
X = X / 255.0  # Normalize
y = to_categorical(df['emotion'], num_classes=7)

# Train-validation split
_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------- Load Existing Model ---------------------
model_path = 'model.h5'
model = load_model(model_path)
print("✅ Model loaded from model.h5")

# --------------------- Confusion Matrix & Report ---------------------
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

print(classification_report(y_true, y_pred_classes, target_names=[
    'Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
    'Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'],
           yticklabels=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# --------------------- Saliency Map ---------------------
def saliency_map(model, image, class_index):
    image = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        loss = predictions[0, class_index]
    grads = tape.gradient(loss, image)
    grads = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    plt.imshow(grads, cmap='jet')
    plt.title('Saliency Map')
    plt.axis('off')
    plt.show()

# Use a valid index after filtering
if len(X_val) > 0:
    saliency_map(model, X_val[0], y_true[0])
else:
    print("No validation data available after filtering for Saliency Map.")


# --------------------- Grad-CAM ---------------------
def grad_cam(model, img, class_index, layer_name='conv2d_5'):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    img = tf.expand_dims(img, axis=0)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize(heatmap, (48,48))
    plt.imshow(heatmap, cmap='jet')
    plt.title('Grad-CAM')
    plt.axis('off')
    plt.show()

# Use a valid index after filtering
if len(X_val) > 0:
    grad_cam(model, X_val[0], y_true[0])
else:
    print("No validation data available after filtering for Grad-CAM.")


# --------------------- LIME Explanation ---------------------
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
import matplotlib.pyplot as plt
import numpy as np

explainer = lime_image.LimeImageExplainer()

img = X_val[0].reshape(48, 48, 1)
target_class = np.argmax(y_val[0])

def lime_predict(images):
    images = np.array(images, dtype=np.float32)
    if images.shape[-1] == 3:
        images = np.mean(images, axis=-1, keepdims=True)
    images /= 255.0
    return model.predict(images)

def segmenter(image):
    return slic(image, n_segments=10, compactness=5, sigma=1)

explanation = explainer.explain_instance(
    image=img.squeeze(),
    classifier_fn=lime_predict,
    top_labels=1,
    hide_color=0,
    num_samples=1000,
    segmentation_fn=segmenter
)

temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=False,
    hide_rest=False
)

# Overlay
plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.title('LIME Explanation')
plt.axis('off')
plt.show()

# Optional: view the raw mask
plt.imshow(mask, cmap='jet')
plt.title('Raw LIME Mask')
plt.axis('off')
plt.show()