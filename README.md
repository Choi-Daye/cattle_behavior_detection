# SOGAETING: AI-Based Cow Mounting Behavior Detection  

**YOLOv10-based intelligent vision system** for detecting cow mounting behavior (estrus period) in real time, helping reduce livestock breeding losses and improving farm management efficiency.

---

## 1. Project Overview  

This project proposes an **AI-based solution** to prevent economic losses caused by undetected **mounting behavior** in cows during their estrus period.  
Since over **65% of mounting occurs at night**, the system provides **real-time detection and alert notifications** to farmers.  

- **AI Model:** YOLOv10 (parameter-tuned high-precision version)  
- **Objective:** Real-time and accurate detection of mounting behavior  
- **Focus:** Robust detection under low light, occlusion, and group interactions  

---

## 2. Key Features  

- **Real-Time Detection**  
  Detects mounting behavior from farm CCTV streams in real time using YOLOv10.

- **Automated Behavior Recognition**  
  Recognizes estrus-related mounting behaviors using deep learning–based object detection  
  combined with multi-frame validation logic.

- **Database Integration**  
  Stores both CCTV footage and detection results for record tracking and retraining.

---

## 3. Model Architecture & Optimization  

**YOLOv10 (Specially Tuned for Mounting Detection)**  
- **Real-time detection** optimized for farm CCTV conditions  
- Parameter tuning targeted at **cow mounting motion and frame continuity**  

**Training Parameters:**

| Epochs | Batch | ImgSz | Mosaic | Mixup | Copy-Paste | HSV-H | HSV-V |
|--------|-------|-------|--------|-------|-------------|-------|-------|
| 50 | 32 | 640 | 0.5 | 0.1 | 0.5 | 0.05 | 0.6 |


**Class Labeling:**  
- **Mounting Behavior (Positive)** vs **Normal Behavior (Negative)**
- Label annotations verified using recorded farm videos  

---

### 3.1 Post-Processing Pipeline  

After raw YOLOv10 detection, multi-stage filtering was applied to ensure temporal consistency:

| Stage | Condition | Description |
|-------|------------|--------------|
| **1. Confidence Filter** | ≥ **0.75** | Only high-confidence detections kept |
| **2. Height Difference (Δy)** | ≥ **10 px** | Detect vertical pixel displacement indicating mounting |
| **3. Temporal Consistency** | > **5 consecutive frames** | Ensures sustained action before alert |
| **4. Frame Gap** | ≤ **30 frames** | Groups nearby detections as one event |

**Filter Flow:**
- Raw Detection → Confidence ≥ 0.75 → Height Δ ≥ 10px → Consecutive Frames > 5 → Frame Gap ≤ 30 → ✅ Mounting Confirmed

---

## 4. Dataset  

- **Company-Provided Data**  
  - Labeled mounting behavior dataset  
  - Real-farm CCTV videos (day & night)

---

## 5. Experimental Process  

### 5.1 YOLOv10 Parameter Tuning  

| Parameter | Value | Purpose |
|------------|--------|---------|
| **Epochs** | 50 | Optimal convergence |
| **Batch Size** | 32 | Stable GPU memory usage |
| **Image Size** | 640 | Speed–accuracy balance |
| **Mosaic / Mixup / Copy-Paste** | 0.5 / 0.1 / 0.5 | Data diversity for cow herd motion |
| **HSV-Hue / Value** | 0.05 / 0.6 | Nighttime adaptation enhancement |

**Validation Control:**  
- Early stopping + overfitting monitoring  
- CPU/GPU split validation for consistency  

---

### 5.2 Additional Filtering Conditions  

| Condition | Logic |
|------------|--------|
| **Confidence ≥ 0.75** | Filter low-confidence detections |
| **Height Δ ≥ 10 px** | Capture significant upward motion |
| **Frames > 5** | Confirm sustained contact behavior |
| **Frame gap ≤ 30** | Treat detections within window as one event |

---

## 6. Final Results  

**Performance Metrics (Test Dataset)**  

| Metric | Result | Target | Status |
|--------|--------|---------|--------|
| **Recall (Detection Rate)** | **93.7%** | ≥ 90% | ✅ Achieved |
| **FPR (False Positive Rate)** | **16.6%** | ≤ 20% | ✅ Achieved |
| **Detection Accuracy (Nighttime)** | High Consistency | – | ✅ Verified |

> Model achieved **93.7% recall** and reduced false alarms to **16.6%** using post-validation filters.  
> Most missed events occurred when multiple cows overlapped under poor illumination.

<img width="294" height="217" alt="Image" src="https://github.com/user-attachments/assets/caee7f40-e26d-4fa7-8fe9-b29cf55a18e6" />

---

## 7. Environment  

- **Framework:** YOLOv10 (Ultralytics)  
- **Libraries:** PyTorch, OpenCV, NumPy, Pandas  
- **Hardware:** NVIDIA RTX GPU (CUDA 11.x)  
- **Development Platforms:** VSCode, Docker  

---

## 8. System Integration  

**Architecture Overview:** CCTV → Custom-trained YOLOv10 → Post-filtering → Alert Module → Database Storage (MySQL) → Dashboard Display (Web)

- **Inference:** YOLOv10 `best.pt` (user-trained checkpoint)  
- **Filtering:** Custom confidence/height/frame validation  
- **Storage:** MySQL (timestamps + detection results)  


## 9. Run detection on sample video


