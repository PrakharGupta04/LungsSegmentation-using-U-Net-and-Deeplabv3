# LungsSegmentation-using-U-Net-and-Deeplabv3
A deep learning project for automatic lung segmentation in chest X-rays using U-Net and DeepLabV3. Includes preprocessing, balanced dataset creation, training, and evaluation on Normal, COVID-19, and Pneumonia images from RSNA and NIH. Provides full pipeline, visual outputs, and metric comparison.

# Lung Segmentation using U-Net and DeepLabv3  
### Comparative Segmentation of Chest X-ray Images

This project performs **automatic lung segmentation** from chest X-ray images using two powerful deep learning architectures:

- **U-Net (Custom encoder–decoder)**
- **DeepLabv3-ResNet50 (fine-tuned for 1-channel lung mask output)**

The repository includes **training scripts**, **evaluation pipeline**, **dataset preparation**, **inference utilities**, and a full **Streamlit UI** for interactive mask generation and comparison.

---

##  Features

-  **Custom U-Net implementation**  
-  **DeepLabv3 head finetuning & checkpoint cleaning**
-  **Dataset loader with augmentation**
-  **Evaluation with Dice, IoU, F1, Precision, Recall, Accuracy**
-  **Side-by-side mask comparison (probability maps, overlays)**
-  **Streamlit UI for real-time inference**
-  **Debugging utilities for diagnosing model outputs**
-  **Balanced dataset creation (undersampling + augmentation)**

---

## Project Structure
LungsSeg/
│
├── app_streamlit.py # Streamlit inference UI
├── train_unet.py # Train U-Net model
├── unet_model.py # U-Net architecture
├── finetune_deeplab_head.py # Fine-tune DeepLab head
├── infer_compare.py # Compare UNet & DeepLab on an image
├── evaluate_models.py # Computes metrics + visual results
├── dataset_loader.py # Data augmentation & loading
├── create_balanced_dataset.py # Create balanced dataset
├── preprocess.py # Preprocessing pipeline
├── mask_visualization.py # Visualize masks
├── single_infer_debug.py # Detailed per-image debugging
├── clean_and_check_deeplab_ckpt.py
├── inspect_deeplab_ckpt.py
│
├── requirements.txt # Python dependencies
└── README.md


> **Note:** Dataset, checkpoints, virtual environment folders are intentionally excluded via `.gitignore`.

---

#  Dataset Format

Your dataset must follow this directory structure:
data/
└── train/
├── CLASS_NAME/
│ ├── images/
│ └── masks/
├── CLASS_NAME/
├── images/
└── masks/
└── val/
└── test/


Example class names:  
`Normal`, `COVID`, `Pneumonia`, etc.

Each **image** must have a matching **mask** file with the **same filename**.

### Creating a balanced dataset  
Run:

```bash
python create_balanced_dataset.py \
    --dataset_root raw_dataset \
    --out_root data \
    --target_per_class 500

**#  Training the Models**
Train U-Net
python train_unet.py


This:


loads dataset from data/train/

trains the U-Net model

saves checkpoint inside checkpoints/

Fine-tune DeepLabv3 Head
python finetune_deeplab_head.py


This:

loads DeepLabv3-ResNet50 backbone

modifies classifier for 1-channel output

fine-tunes for segmentation

saves checkpoint into checkpoints_deeplab/


**Evaluate Models (Full Comparison)**

Run:

python evaluate_models.py


This script:

Evaluates U-Net on data/val/

Evaluates DeepLabv3 on datasets_reduced_500/val/ (if provided)

Calculates metrics:
Dice, IoU, F1, Precision, Recall, Accuracy

Saves results into:

results/comparison_<timestamp>/
    ├── comparison_report.json
    ├── summary.json
    ├── metrics_comparison.png
    └── vis_examples/

Run the Streamlit Web UI (Interactive)

This is the most user-friendly way to test your models.

Start the app
streamlit run app_streamlit.py


**Features:**

Upload any X-Ray image

Get U-Net & DeepLabv3 predicted lung masks

Interactive threshold sliders

Probability maps

Binary masks

Color overlays

Distribution plots

JSON summaries

**Single Image Inference (UNet vs DeepLab)**

Compare masks from both models:

python infer_compare.py --image path/to/xray.png


Debug DeepLab outputs:

python single_infer_debug.py --image path/to/file.png

 **How to Run This Project (FULL GUIDE)**
 Step 1: Clone the Repository
git clone https://github.com/PrakharGupta04/LungsSegmentation-using-U-Net-and-Deeplabv3.git
cd LungsSegmentation-using-U-Net-and-Deeplabv3

Step 2: Create Virtual Environment
python -m venv lungseg


Activate:

Windows

lungseg\Scripts\activate


Linux/Mac

source lungseg/bin/activate

Step 3: Install Dependencies
pip install -r requirements.txt

Step 4: Prepare Dataset

Place your dataset inside:

data/train/
data/val/
data/test/


Or run the auto-balancing script.

Step 5: Run Training / Evaluation / Streamlit UI
Task	Command
Train U-Net	python train_unet.py
Train DeepLab	python finetune_deeplab_head.py
Evaluate both models	python evaluate_models.py
Streamlit app	streamlit run app_streamlit.py
Single image inference	python infer_compare.py --image <file>

**Metrics Explained**
Metric	Meaning
Dice Score	Measures overlap (best for segmentation)
IoU	Intersection over Union of predicted vs real mask
F1 Score	Balance between Precision and Recall
Precision	How many predicted lung pixels are correct
Recall	How many actual lung pixels are detected
Accuracy	Percentage of correct pixels (not ideal for segmentation)


** Future Enhancements**

Add UNet++ / Attention U-Net

Apply CRF post-processing

Export to ONNX / TFLite for deployment

Integrate Grad-CAM for mask interpretability

 **Author

Prakhar Gupta
B.Tech | Machine Learning | Deep Learning
GitHub: https://github.com/PrakharGupta04**

