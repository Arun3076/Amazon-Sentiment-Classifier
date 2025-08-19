🛒 Amazon Sentiment Classifier
This project is a Transformer-based sentiment classifier built for analyzing Amazon product reviews. It predicts whether a review is positive, negative, or neutral using Hugging Face Transformers.

📂 Project Structure
Amazon-Sentiment-Classifier/
│── app.py                        # Main application script
│── Model.ipynb                   # Jupyter Notebook (training + testing)
│── amazon_reviews_full.csv/xlsx  # Amazon reviews dataset
│── amazon-model/                 # (Not uploaded, too large for GitHub)
│── README.md                     # Documentation


⚡ Features
Uses Hugging Face Transformers for text classification.
Trained on the Amazon Reviews dataset.
app.py allows quick sentiment predictions on custom input reviews.
Jupyter Notebook (Model.ipynb) shows full training + evaluation pipeline.


🚀 How to Run
Since the fine-tuned model is very large (~255 MB), it is not uploaded to GitHub.
The notebook will automatically download the base model from Hugging Face when you run it.

1️⃣ Clone the repository
git clone https://github.com/your-username/Amazon-Sentiment-Classifier.git
cd Amazon-Sentiment-Classifier

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Jupyter Notebook
Open and execute all cells:
jupyter notebook Model.ipynb
The first run will automatically download the Hugging Face model.
The model will be cached locally (~/.cache/huggingface/transformers).

4️⃣ Run the App Script
python app.py
Enter any Amazon product review text, and the model will return the sentiment.

📌 Notes
The fine-tuned amazon-model folder is not included due to GitHub file size limits.
Hugging Face will handle downloading the base pre-trained model automatically.
If you want to share your trained weights, consider uploading them to Hugging Face Hub or Google Drive.
