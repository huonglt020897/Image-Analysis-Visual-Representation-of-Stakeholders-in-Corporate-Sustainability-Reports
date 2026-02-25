# Who Gets Seen? Visual Representation of Stakeholders in Corporate Sustainability Reports

> *A computer vision analysis of human representation across 156 CSR reports from conglomerate companies*

---

## 📌 Overview

Corporate Sustainability Reports (CSRs) are not just text — they communicate values visually. This project investigates **who appears in the imagery** of CSR reports published by conglomerates: examining demographic characteristics and stakeholder types to surface patterns in visual representation.

**Research Questions:**
1. Who gets represented? What are their demographic characteristics (gender, apparent age)?
2. Which stakeholder groups (employees, leadership, communities) appear in CSR imagery — and with what frequency?

---

## 🗂️ Dataset

| Metric | Value |
|--------|-------|
| CSR reports analyzed | 156 |
| Reports containing images | 151 |
| Total images extracted | 12,862 |
| Images with humans detected | 4,780 |
| Humans detected (total) | 24,455 |
| Faces analyzed | 12,439 |

Reports span multiple years and regions, with company size ranging from large-cap to mid-cap conglomerates.

---

## 🔧 Technical Pipeline

The analysis runs a **three-stage computer vision pipeline** on each extracted image:

```
PDF Reports
    │
    ▼
[PyMuPDF] Image Extraction
    │
    ▼
┌───────────────────────────────────────────┐
│           Image Analysis Pipeline         │
│                                           │
│  Step 1: YOLOv8 — Human Detection        │
│    └─ Count people per image              │
│    └─ Classify group size                 │
│                                           │
│  Step 2: MTCNN + OpenCV DNN               │
│          Face Analysis                    │
│    └─ Detect faces                        │
│    └─ Classify gender (Male/Female)       │
│    └─ Estimate age group                  │
│    └─ Identify portrait composition       │
│                                           │
│  Step 3: YOLOv8 — Object Detection       │
│    └─ Safety/industrial equipment         │
│    └─ Office items                        │
│    └─ Stakeholder type classification     │
└───────────────────────────────────────────┘
    │
    ▼
Structured Results (CSV) + Visualizations
```

### Classification Logic

**Group Size:**
| Category | People Count |
|----------|-------------|
| Single Person | 1 |
| Small Group | 2–5 |
| Medium Group | 6–15 |
| Large Group | 15+ |

**Stakeholder Type (rule-based):**
| Category | Detection Signal |
|----------|-----------------|
| Leadership/Executives | Portrait composition (single person, face >10% of frame, centered) + middle-age/senior |
| Frontline/Production Workers | Safety equipment OR industrial machinery detected |
| Office/Knowledge Workers | Office items (laptop, monitor, desk, etc.) detected |
| Mixed/Ambiguous | Insufficient context signals |

**Demographic Attributes:**
- **Gender:** Male / Female / Unknown (via OpenCV DNN `gender_net.caffemodel`)
- **Age Groups:** Youth (<25) / Young Adult (25–40) / Middle Age (40–60) / Senior (60+) (via `age_net.caffemodel`)

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `PyMuPDF (fitz)` | Extract images from PDF reports |
| `YOLOv8 (ultralytics)` | Human detection & object classification |
| `MTCNN` | Multi-task face detection |
| `OpenCV DNN` | Age & gender prediction (Caffe models) |
| `pandas` | Data wrangling |
| `matplotlib` / `seaborn` | Visualization |
| `pyreadr` | Read R `.rds` metadata file |

---

## 📊 Key Findings

### Data Coverage
- Images were extracted from the majority of reports; a meaningful share contained at least one detectable human figure.
- Face detection was successful for roughly half of all detected humans, reflecting real-world variation in image angle, resolution, and composition.

### Group Size Distribution
Small groups (2–5 people) are the most prevalent category, followed by single-person images  — suggesting CSR visuals lean toward individual representation rather than collective scenes.

### Stakeholder Representation
The majority of human-containing images fall into the **Mixed/Ambiguous** category, followed by **Office/Knowledge Workers** — indicating that executive portraiture is a prominent visual convention in CSR communication.

### Gender & Age
- **Gender:** Female faces substantially outnumber male faces across the dataset.
- **Age:** Young adult and middle-age individuals dominate; youth and seniors are comparatively underrepresented.

---

## ⚙️ Setup & Usage

### Prerequisites

```bash
pip install opencv-python ultralytics mtcnn pyreadr pymupdf pandas polars matplotlib seaborn tqdm
```

OpenCV age/gender models are downloaded automatically on first run via `download_opencv_models()`.

### Running the Pipeline

1. Place CSR PDFs in your configured `dir_pdf` directory.
2. Open `image_human_analysis.ipynb` and run cells sequentially.
3. Image extraction runs once and saves metadata to `extracted_images_metadata.csv`.
4. The analysis pipeline (`analyze_images_and_save`) processes in configurable batches (default: 500 images) with automatic MTCNN memory refresh between batches.
5. Final results are saved to `image_analysis_results.csv`.

> **Note:** Image extraction and analysis are designed to be run once. Subsequent runs load from saved CSVs.

---

## ⚠️ Limitations

- **Gender classification** is binary (Male/Female) due to model constraints; the "Unknown" category captures ambiguous cases.
- **Age estimation** relies on an 8-bracket Caffe model; predictions carry inherent uncertainty especially at age group boundaries.
- **Stakeholder classification** is rule-based and context-dependent — the "Mixed/Ambiguous" category is expected to be large given limited visual context in many CSR images.
- Face detection performance degrades for small faces, non-frontal angles, and low-resolution images.
- Findings are descriptive and reflect *apparent* visual presentation, not verified identity data.

---

## 📄 Data Source

- GRI report metadata sourced from an `.rds` file (`DAV_assignment.rds`) covering conglomerate-sector companies reporting under the Global Reporting Initiative (GRI) framework.
- GRI pdf reports.

