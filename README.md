🌾 Agri-EfficientNet
Lightweight Deep Learning Model for Plant Disease Detection

🧠 Overview
Agri-EfficientNet is a lightweight CNN model for identifying crop leaf diseases with high accuracy.
It is currently locally deployed via Flask as a web demo, allowing users to upload images and receive instant diagnoses.

⚙️ Key Features

🎯 99% classification accuracy on the PlantVillage dataset
🧩 Lesion-Focused Attention (LFA) for subtle disease detection
💻 Local Flask demo for real-time prediction on any computer
🧠 EfficientNet-based architecture — smaller and faster than ResNet-50

📊 Model Summary
Metric	Agri-EfficientNet	ResNet-50
Accuracy	99.0%	98.0%
Model Size	15.8 MB	94.1 MB
CPU Inference	22 ms	60 ms

🧾 Dataset
Trained on the PlantVillage Dataset
with over 50,000 labeled leaf images across 38 crop disease classes.

💻 Run Locally
# Install dependencies
pip install -r requirements.txt
# Run Flask app
python app.py
Then open your browser at 👉 http://127.0.0.1:5000/

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
