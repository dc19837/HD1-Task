import pandas as pd # importing pandas library
import re # importing regular expression library
from bs4 import BeautifulSoup # importing BeautifulSoup library for parsing HTML and XML documents

from sklearn.feature_extraction.text import TfidfVectorizer # importing TfidfVectorizer for text feature extraction
from sklearn.model_selection import train_test_split # importing train_test_split for splitting data into training and testing sets
from sklearn.linear_model import LogisticRegression # importing LogisticRegression for classification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # importing metrics for evaluation

import nltk # importing nltk for natural language processing
from nltk.corpus import stopwords # importing stopwords from nltk

import matplotlib.pyplot as plt # importing matplotlib for plotting
import seaborn as sns # importing seaborn for statistical data visualization
 
def get_stopwords(): # function to get the stopwords
    try: # try to get the stopwords
        sw = stopwords.words('english') # getting the stopwords
    except LookupError: # if the stopwords are not available
        nltk.download('stopwords') # downloading the stopwords
        sw = stopwords.words('english') # getting the stopwords again
    return sw # returning the stopwords

def clean_text(text): # function to clean the text data
    """
    Cleans the given text by:
    - Removing HTML tags using BeautifulSoup.
    - Removing non-alphanumeric characters.
    - Converting text to lowercase.
    - Removing extra whitespace.
    """
    soup = BeautifulSoup(text, "html.parser")  # Parse and remove HTML tags
    cleaned_text = soup.get_text(separator=" ")  # Get only text content from HTML
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', cleaned_text)  # Remove special characters, keep letters/numbers/spaces
    cleaned_text = cleaned_text.lower()  # Convert text to lowercase
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Replace multiple spaces with single space and trim
    return cleaned_text  # Return the cleaned text

def load_and_preprocess_data(file_path): # function to load and preprocess the data
    """
    Loads and preprocesses the dataset from the given CSV file.
    Expects the CSV to contain the following columns:
      sender, receiver, date, subject, body, label, urls
      
    Combines subject, body, and urls into a single text field named 'email_content'.
    All text fields (subject, body, urls) are converted to strings.
    """
    df = pd.read_csv(file_path) # loading the CSV file into a pandas DataFrame
    required_columns = ['sender', 'receiver', 'date', 'subject', 'body', 'label', 'urls'] # list of required columns
    for col in required_columns: # checking each required column
        if col not in df.columns: # if any column is missing
            raise ValueError(f"The CSV file must contain the column '{col}'.") # raise error

    for col in ['subject', 'body', 'urls']: # processing columns used for content
        df[col] = df[col].fillna("").astype(str) # replacing NaNs with empty strings and ensuring text format
    df['email_content'] = df['subject'] + " " + df['body'] + " " + df['urls'] # combining text fields into one
    df['clean_content'] = df['email_content'].apply(clean_text) # applying cleaning to the combined text
    return df # returning the processed DataFrame

def extract_features(text_data): # Function to convert text data to numerical TF-IDF features
    """
    Converts the cleaned text data into a TF-IDF feature matrix.
    Limits to the top 5000 features and removes English stopwords.
    Uses NLTK stopwords with a fallback to ensure resource avaliability.
    """
    sw = get_stopwords() # getting the stopwords
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words=sw) # creating the TfidfVectorizer object
    X = tfidf_vectorizer.fit_transform(text_data) # fitting and transforming the text data
    return X, tfidf_vectorizer # return the feature matrix and the vectorizer object

def plot_confusion_matrix(cm): # function to plot the confusion matrix
    """
    Plots the confusion matrix using matplotlib heatmap.
    """
    plt.figure(figsize=(8, 6)) # setting the figure size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing']) # plotting the heatmap
    plt.xlabel('Predicted Label') # setting the x-axis label
    plt.ylabel('Actual Label') # setting the y-axis label
    plt.title('Confusion Matrix Heatmap') # setting the title
    plt.show() # showing the plot

def train_and_evaluate(X, y):
    """
    Splits the data into training and test sets, trains a Logistic Regression model,
    and evaluates the model's performance.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # spliting the data into training and testing sets
    model = LogisticRegression(max_iter=1000, solver='lbfgs') # creating the Logistic Regression model
    model.fit(X_train, y_train) # fitting the model to the training data
    y_pred = model.predict(X_test) # predicting the labels for the test data
    accuracy = accuracy_score(y_test, y_pred) # calculating the accuracy
    print(f"Accuracy: {accuracy * 100:.2f}%\n") # printing the accuracy
    print("Classification Report") # printing the classification report
    print(classification_report(y_test, y_pred)) # printing the classification report, recall and f1-score

    cm = confusion_matrix(y_test, y_pred) # calculating the confusion matrix
    print("Confusion Matrix") # printing the confusion matrix
    print(cm) # printing the confusion matrix
    plot_confusion_matrix(cm) # plotting the confusion matrix
    return model # return the trained model

def main(): # main function to run the script
    file_path = 'CEAS_08.csv' # path to the CSV file
    print("Loading and preprocessing dataset...") # loading and preprocessing the data
    df = load_and_preprocess_data(file_path) # loading and preprocessing the data
    print("Dataset loaded successfully with 39154 samples.\n") # dataset loaded and preprocessed
    print("Extracting TF-IDF features...") # extracting the TF-IDF features
    X, tfidf_vectorizer = extract_features(df['clean_content']) # extracting the TF-IDF features
    y = df['label'] # getting the labels
    print("Training and evaluating the model...") # TF-IDF features extracted successfully
    model = train_and_evaluate(X, y) # training and evaluating the model

if __name__ == "__main__": # if the script is run directly
    main() # call the main function