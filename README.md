🌾 Agri-EfficientNet: A Data-Efficient Deep Learning Framework

A lightweight, data-efficient deep learning framework for robust plant disease diagnosis, especially in data-scarce scenarios.

This repository contains the official code and dataset for the paper: "Agri-EfficientNet: A Data-Efficient Deep Learning Framework for Robust Diagnosis of Durian Diseases in Complex Field Environments."

🧠 Overview
This project tackles the "lab-to-field" gap in agricultural AI. While most models perform well on clean datasets (like PlantVillage), they fail in real-world field conditions.

More importantly, we address the critical data scarcity challenge for high-value, under-represented crops. This framework introduces 8 novel disease categories for Durian (e.g., Pink Disease, Leafhopper), built from rare, expert-verified "seed" images.

The model is locally deployed via Flask as a web demo, allowing users to upload images and receive instant diagnoses on a local computer.

⚙️ Key Features

🌱 First-of-its-Kind Durian Data: Introduces 8 novel, field-verified Durian disease categories, addressing a critical data gap.

🧩 Lesion Focused Attention (LFA): A novel attention module that mitigates background noise (hands, soil) and prevents overfitting in low-data regimes.

🎯 High Robustness: Achieves a 94.6% Macro F1-Score on our challenging hybrid field benchmark, significantly outperforming ResNet-50.

⚡ Lightweight & Fast: 6x smaller (15.8 MB) and 3x faster on CPU (21.18 ms) than ResNet-50, proving its feasibility for practical deployment.

💻 Local Flask Demo: A simple web interface to run real-time predictions on your local machine.

📊 Model Summary (on Hybrid Field Benchmark)
This table reflects the performance on our challenging, 46-class hybrid test set (8,160 images), not the clean PlantVillage dataset.
<img width="3613" height="637" alt="model_performance_table" src="https://github.com/user-attachments/assets/5afa2c96-a3c2-4e44-9394-771acdb4d787" />

🧾 Dataset: The 46-Class Hybrid Benchmark
This is not just the PlantVillage dataset. Our model is trained on a comprehensive "hybrid benchmark" (54,256 images, 46 classes) composed of:

Public Data (38 Classes): The standard PlantVillage dataset.

Novel Field Data (8 Classes): Our 8 new Durian disease categories, built from rare "seed" images and propagated via augmentation.

The full dataset, including our augmented Durian classes, is publicly available:

Dataset Download (Google Drive): https://drive.google.com/file/d/1Ps1JDNoY1dEhswuDiUtGcOyg23Jb6Qa1/view?usp=drive_link

💻 Run Locally

Clone the repository:

git clone [https://github.com/xiaolin200206/Agri-EfficientNet.git](https://github.com/xiaolin200206/Agri-EfficientNet.git)
cd Agri-EfficientNet


Install dependencies:

pip install -r requirements.txt


Run the Flask application:

python app.py


Open your browser and go to http://127.0.0.1:5000 to upload an image.

Citation
If you find this work or dataset useful in your research, please cite our paper (link to be added upon publication).

🚧 Future Work
Integrate with edge hardware (e.g., Raspberry Pi, Jetson Nano)
Add real-time camera input and offline prediction mode
Expand dataset with local farm images for regional adaptation

📘 License
Apache License 2.0 — free for academic and research use with attribution.

🤝 Author
Developed independently by Lin Dingshan
📧 l1055505011@gmail.com
🔗https://github.com/xiaolin200206/Agri-EfficientNet/tree/main
