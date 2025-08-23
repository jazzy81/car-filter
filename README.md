# Car Image Classification & Profile Detection

## 📌 Project Overview

This project focuses on building a multi-stage deep learning pipeline to classify and analyze car images uploaded by users. The primary goal is to ensure that the uploaded image:

1. Represents a **complete car** (not partial).
2. Is captured from a **close view** rather than from a distance.
3. Identifies the **car’s viewing angle** (front, back, side, front-left, front-right, back-left, back-right).
4. Determines the **side profile** (driver side vs passenger side) using mirror detection.

---

## 🧠 Model Development

### 1. **Car Completeness & Distance Classification**

* **Notebook:** `car_filter_yolov8_Classification.ipynb`

* **Task:** Detect whether the uploaded car image is:

  * Complete or partial
  * Captured closely or from a far distance

* **Challenge:**
  Collecting enough **far-distance car images** was difficult, which caused a **class imbalance problem**.

* **Solution:**

  * Used **cost-sensitive learning** by introducing **class weights**.
  * Since YOLOv8 does not natively support class weights, the source code was modified to include a **class weight parameter** in the **Binary Cross Entropy (BCE) loss function**.
  * Training with modified code was done in: `custom_yolo.ipynb`.

---

### 2. **Car Angle Classification**

* **Notebook:** `angle_wise_Detection.ipynb`
* **Task:** Identify the **angle of the car** in the uploaded image:

  * Front view
  * Back view
  * Side view
  * Front-left, Front-right
  * Back-left, Back-right

---

### 3. **Driver vs Passenger Side Profile Detection**

* **Problem:**
  Distinguishing between **driver side** and **passenger side** profiles is extremely challenging because both sides appear almost identical in car images. Models like YOLOv8 or U-Net cannot reliably differentiate them.

* **Solution:**

  * Trained a **mirror detection model** (`mirror_Detection.ipynb`).
  * The model provides the **center coordinates of the car side mirror**.
  * Using these coordinates, a Python script measures the distance of the mirror to the image corners:

    * If the mirror is **closer to the bottom-left corner (0,0)** → Driver side profile
    * If the mirror is **closer to the bottom-right corner (1,1)** → Passenger side profile

---

## ⚙️ Workflow Summary

1. **Upload car image**
2. **Car filter model** (`car_filter_yolov8_Classification.ipynb`) ensures the image is:

   * Complete car
   * Close view
3. **Angle classification model** (`angle_wise_Detection.ipynb`) identifies the car’s angle/profile
4. **Mirror detection model** (`mirror_Detection.ipynb`) determines whether the side profile is **driver** or **passenger**

---

## 🚀 Key Contributions

* Modified YOLOv8 source code to support **class weights** for imbalanced classes.
* Built a **two-step classification system** (car completeness + angle detection).
* Designed a novel approach for **driver vs passenger side detection** using **mirror localization and distance-based heuristics**.


