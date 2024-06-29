![download (2)](https://github.com/DhakshanaMoorthyRDM/Hate-Tweet-Detector/assets/121345776/810c6314-9060-4aec-87e3-9b9f34dd0f93)
# Hate-Tweet-Detector

This Python project analyzes Twitter data to identify hate speech and creates a machine learning model based on the dataset. The dataset is available on Kaggle and contains labels indicating whether the tweets are hate speech or not. Label 1 indicates hate speech, while label 0 indicates non-hate speech tweets.

## Libraries Used:

This project utilizes the following libraries:

•	Pandas: For data manipulation and analysis

•	NumPy: For numerical computations

•	re: For regular expressions and text cleaning

•	Seaborn: For data visualization

•	Matplotlib: For plotting and data visualization

o	style.use('ggplot'): For setting the plotting style

•	NLTK (Natural Language Toolkit): For text processing

o	word_tokenize: For tokenizing text

o	WordNetLemmatizer: For lemmatizing words

o	stopwords: For removing stop words

•	WordCloud: For generating word clouds

•	Scikit-learn: For machine learning

o	TfidfVectorizer: For transforming text data into TF-IDF features

o	train_test_split: For splitting the dataset into training and testing sets

o	LogisticRegression: For building the logistic regression model

o	accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay: For evaluating the model

## Project Overview

The goal of this project is to develop a machine learning model that can accurately detect hate speech in tweets. The main steps involved are:

1.	Data Loading: Load the dataset from Kaggle.

2.	Data Cleaning: Clean the tweets using regular expressions to remove unwanted characters, links, and symbols.

3.	Text Preprocessing: Tokenize the tweets, remove stop words, and lemmatize the words.

4.	Feature Extraction: Convert the text data into numerical features using TF-IDF vectorization.

5.	Model Training: Split the dataset into training and testing sets, then train a logistic regression model on the training data.

6.	Model Evaluation: Evaluate the model's performance using accuracy, classification report, and confusion matrix.

## Folder Structure

```bash
Hate-Tweet-Detector/
│
├── dataset/
│   └── <dataset files>
│
├── Hate Tweet Detector.ipynb
├── Hate Tweet Detector.py
└── README.md
```

## Getting Started

To get started with the project, follow these steps:

1.	Clone the repository:
```bash
git clone https://github.com/yourusername/Hate-Tweet-Detector.git
cd Hate-Tweet-Detector
```
2.	Install the required libraries:
```bash
pip install pandas numpy seaborn matplotlib nltk wordcloud scikit-learn
```
3.	Download the dataset from Kaggle and place it in the dataset directory.
4.	Run the Jupyter Notebook to preprocess the data, train the model, and evaluate its performance:
```bash
jupyter notebook "Hate Tweet Detector.ipynb"
```
Alternatively, you can run the Python script:
```bash
python "Hate Tweet Detector.py"
```

## Acknowledgements
•	Kaggle for providing the dataset.

•	The developers and maintainers of the libraries used in this project.


Feel free to explore, contribute, and provide feedback!

