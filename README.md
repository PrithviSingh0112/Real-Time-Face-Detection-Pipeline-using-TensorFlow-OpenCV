
# 🧠 Real-Time Face Detection Pipeline using TensorFlow & OpenCV

This project implements a deep learning–based **face detection pipeline** using TensorFlow, OpenCV, and a custom-labeled dataset. It includes dataset annotation, augmentation, model training, and real-time inference—all in one place.

---

## 🚀 Pipeline Overview

```
📁 Data Collection → 🏷️ Annotation → 🧪 Augmentation → 🧠 Model Training → 🎥 Real-Time Detection
```

---

## 🛠️ Tools & Libraries Used

| Tool | Purpose |
|------|---------|
| **TensorFlow** | Deep learning framework for model creation and training |
| **OpenCV** | Image & video handling, drawing bounding boxes |
| **LabelMe** | Annotation of face bounding boxes in images |
| **Albumentations** | Fast & flexible image augmentations |
| **Matplotlib** | Visualization of training history and detection results |

---

## 📂 Dataset Creation

1. **Image Collection**:
   ```python
   cap = cv2.VideoCapture(0)
   while cap.isOpened():
       ret, frame = cap.read()
       imgname = os.path.join(IMG_PATH, f'{str(uuid.uuid1())}.jpg')
       cv2.imwrite(imgname, frame)
   ```
   Captures real-time webcam images using OpenCV.

2. **Annotation**:
   - Launch LabelMe with:
     ```bash
     labelme data\images
     ```
   - Draw bounding boxes and export annotations as `.json`.

3. **Organizing Labels**:
   ```python
   for folder in ['train', 'test', 'val']:
       for file in os.listdir(os.path.join('data', folder, 'images')):
           json_name = file.replace('.jpg', '.json')
           src = os.path.join('data', 'labels', json_name)
           dst = os.path.join('data', folder, 'labels', json_name)
           os.replace(src, dst)
   ```

4. **Augmentation (Albumentations)**:
   ```python
   import albumentations as A

   transform = A.Compose([
       A.HorizontalFlip(p=0.5),
       A.RandomBrightnessContrast(p=0.2),
       A.Blur(blur_limit=3, p=0.1)
   ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
   ```

---

## 🧠 Model Architecture

Uses a **VGG16 backbone** for feature extraction and a custom regression head for bounding box prediction.

```python
base_model = VGG16(include_top=False, input_shape=(120, 120, 3))
x = base_model.output
x = GlobalMaxPooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dense(4)(x)  # Bounding box coordinates

model = Model(inputs=base_model.input, outputs=x)
```

- **Loss Function**: Mean Squared Error
- **Optimizer**: Adam

---

## 🏋️ Training the Model

```python
hist = model.fit(train_dataset,
                 epochs=40,
                 validation_data=val_dataset,
                 callbacks=[tensorboard_callback])
```

Visualize training progress:
```python
plt.plot(hist.history['loss'], label='Train Loss')
plt.plot(hist.history['val_loss'], label='Val Loss')
plt.legend()
```

---

## 🎥 Real-Time Inference

```python
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    resized = tf.image.resize(frame, (120, 120)) / 255.0
    yhat = model.predict(tf.expand_dims(resized, 0))

    if yhat[0][0] > 0.5:
        cv2.rectangle(frame,
                      tuple(np.multiply(yhat[1][0][:2], [frame.shape[1], frame.shape[0]]).astype(int)),
                      tuple(np.multiply(yhat[1][0][2:], [frame.shape[1], frame.shape[0]]).astype(int)),
                      (255, 0, 0), 2)
    cv2.imshow('Face Detection', frame)
```

---

## 📈 Sample Results

![sample](assets/sample_detection.jpg)

---

## 📁 Folder Structure

```
FaceDetection/
│
├── data/
│   ├── images/         # Collected images
│   ├── labels/         # LabelMe annotations
│   ├── train/test/val/ # Organized subsets
│
├── model/              # Trained model weights
├── notebooks/          # Jupyter notebooks
├── assets/             # Sample results
└── FaceDetection.ipynb # Main notebook
```

---

## 📝 License

This project is released under the MIT License.
