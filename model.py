#%%
#import the libraries for the system's design 
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import label_binarize

#%%# 
# Load spaCy model
nlp = spacy.load('en_core_web_sm')

#%%
# Load the patient's drug feedback data
df_train_data = pd.read_csv("C:/Users/andre/Downloads/drugsComTrain_raw.csv")
df_test_data = pd.read_csv("C:/Users/andre/Downloads/drugsComTest_raw.csv")

# %%
#to describe the dataset
df_train_data.describe()
df_test_data.describe()

#%%
#to display the dataset's features with 10 records
df_train_data.head(10)
df_test_data.head(10)

# %%
#To combine the train and test dataset and check the merged data shape
merge = [df_train_data,df_test_data]
df_review_data= pd.concat(merge,ignore_index=True)

df_review_data.shape   

#%%
#To describe and get detailed info on the merged dataset
df_review_data.head()
df_review_data.describe()
df_review_data.info()
df_review_data.count()

#%%
#Preprocessing of the Dataset
#checking the null values and features with the highest number of null values
df_review_data.isnull().sum()

#%%
# dropping the null values
df_review_data.dropna(inplace=True, axis=0)

# %%
# A function to preprocess the patient's review
def preprocess_text(text):
    # text = re.sub(r"I&#039;", "", text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    doc = nlp(text.lower())
    text = text.replace("n't", " not")
    return ' '.join(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)

#%%
#Applying the function on the review feature
df_review_data['cleaned_feedback'] = df_review_data['review'].apply(preprocess_text)

#%%
#To check if the cleaned feadback has been added to the dataset as a new feature
df_review_data.head(30)

#%%
#Performing of sentiment analysis using Textblob to categorize the sentiments
# function to apply sentiment analysis with TextBlob and return the polarity
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

#%%
#Applying the function on the patient's cleaned review
df_review_data[['polarity', 'subjectivity_score']] = df_review_data['cleaned_feedback'].apply(lambda text: pd.Series(analyze_sentiment(text)))

# %%
#To test and check if the polarity and subjectivity score features has been added to the dataset.
df_review_data.head(20)

#%%
# function to categorize the sentiment based on the polarity
def categorize_sentiment(polarity):
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

#%%
df_review_data['sentiment'] = df_review_data['polarity'].apply(categorize_sentiment)

#%%
df_review_data.head()

#%%
#To test and check the added new feature on the dataset
df_review_data.head(10)

#%%
#Exploratory and Visualization of the result of the performed sentiment analysis on the dataset
# Plot the distribution of sentiment scores
plt.figure(figsize=(10, 6))
sns.histplot(df_review_data['polarity'], bins=30, kde=True)
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

#%%
# Plot the distribution of sentiment categories
plt.figure(figsize=(10, 6))
sns.countplot(data=df_review_data, x='sentiment')
plt.title('Distribution of Sentiment Categories')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# %%
# Correlating between specific drugs and sentiment
drug_sentiment = df_review_data.groupby('drugName')['polarity'].mean().reset_index()
drug_sentiment = drug_sentiment.sort_values(by='polarity', ascending=False)

plt.figure(figsize=(15, 10))
sns.barplot(data=drug_sentiment, x='polarity', y='drugName')
plt.title('Average Sentiment Score by Drug')
plt.xlabel('Average Sentiment Score')
plt.ylabel('Drug')
plt.show()

#%%
# Positive and Negative feedback Word Clouds Generation and plots
positive_feedback = ' '.join(df_review_data[df_review_data['sentiment'] == 'Positive']['cleaned_feedback'])
negative_feedback = ' '.join(df_review_data[df_review_data['sentiment'] == 'Negative']['cleaned_feedback'])

positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_feedback)
negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_feedback)

# Plot the word clouds
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Positive Feedback Word Cloud')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Negative Feedback Word Cloud')
plt.axis('off')

plt.show()

#%%
# function to analyze common themes in feedback
def get_common_words(corpus, n=10):
    vec = CountVectorizer(stop_words='english').fit(corpus)
    wordsBag = vec.transform(corpus)
    wordsSum = wordsBag.sum(axis=0) 
    wordsFreq = [(word, wordsSum[0, idx]) for word, idx in vec.vocabulary_.items()]
    wordsFreq = sorted(wordsFreq, key = lambda x: x[1], reverse=True)
    return wordsFreq[:n]

#applying the get function on the sentiment and the pre-processed review
positive_common_words = get_common_words(df_review_data[df_review_data['sentiment'] == 'Positive']['cleaned_feedback'])
negative_common_words = get_common_words(df_review_data[df_review_data['sentiment'] == 'Negative']['cleaned_feedback'])

#printing the top 10 positive and negative feedback
print("Top 10 words in positive feedback:")
print(positive_common_words)

print("\nTop 10 words in negative feedback:")
print(negative_common_words)

#%%
# Aspect Extraction and Classification
#declaring of the aspects
aspects = ['effectiveness', 'side effects', 'cost', 'experience']

#A function to extract aspect 
def extract_aspects(text):
    doc = nlp(text)
    aspects = set() 
    for chunk in doc.noun_chunks:
        aspect = chunk.text.strip()
        if len(aspect) > 1: 
            aspects.add(aspect)
    return list(aspects)

# Applying aspect extraction on the feedback feature
df_review_data['aspects'] = df_review_data['cleaned_feedback'].apply(extract_aspects)

#Classifying the aspects based on the extracted aspects
def classify_aspects(aspects):
    aspect_sentiments = {}
    for aspect in set(aspects):  
        sentiment = TextBlob(aspect).sentiment.polarity
        if sentiment > 0:
            sentiment_label = 'positive'
        elif sentiment < 0:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        aspect_sentiments[aspect] = sentiment_label
    
    return aspect_sentiments

# Apply aspect classification
df_review_data['aspect_classification'] = df_review_data['aspects'].apply(classify_aspects)

# Print relevant columns
print(df_review_data[['review', 'aspects', 'aspect_classification']])

#%%
df_review_data.head(20)

#%%
##Checking for unbalanced feature
#functions to calculate, visualize and print the distribution of classes

def calculate_distribution(df_review_data, sentiments):
    return df_review_data[sentiments].value_counts(normalize=True)

def visualize_distribution(distribution, sentiments):
    plt.figure(figsize=(10, 6))
    distribution.plot(kind='bar')
    plt.title(f'Distribution of {sentiments}')
    plt.xlabel(sentiments)
    plt.ylabel('Frequency')
    plt.show()

def main():
   
    distribution = calculate_distribution(df_review_data, 'sentiments')
    visualize_distribution(distribution, 'sentiments')
    print(distribution)

if __name__ == "__main__":
    main()

#%%
#to print the class distribution
class_distribution = df_review_data['sentiments'].value_counts()
print("Class distribution using value_counts:")
print(class_distribution)
#%%
#to check class distribution using Counter
class_distribution_counter = Counter(df_review_data['sentiments'])
print("\nClass distribution using Counter:")
print(class_distribution_counter)

#to visualize class distribution using matplotlib
plt.figure(figsize=(8, 6))
class_distribution.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Class Distribution')
plt.show()

#to visualize class distribution using seaborn
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiments', data=df_review_data)
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Class Distribution')
plt.show()

#%%
#to perform feature engineering using tfidfvectorizer
vectorizer = TfidfVectorizer()
df_review = vectorizer.fit_transform(df_review_data['cleaned_feedback'])
df_review.shape

#%%
#to apply class on the rating feature (label)
df_review_data['sentiments'] = df_review_data["rating"].apply(lambda x: 1 if x > 5 else 0)

#%%
#to save the sentiments column with a variable
sentiments = df_review_data['sentiments']

#%%
#to split the dataset into train and test sets
X_train,X_test,Y_train,Y_test = train_test_split(df_review,sentiments,test_size=0.33,random_state=42)

#%%
#to balance the unbalanced classes
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, Y_train)

# to check the distribution of classes after applying SMOTE
print(f"Original class distribution: {Counter(Y_train)}")
print(f"Resampled class distribution: {Counter(y_resampled)}")

#%%
#resaving the sampled data
X_train_main = X_resampled
Y_train_main = y_resampled

#%%
# a class to implement multinomial naive bayes
class MultinomialNaiveBayes(BaseEstimator, ClassifierMixin):
    #function to initialize
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    #function fit to calulate the prior and likelihood
    def fit(self, X_train, Y_train):
        noOfDocs, noOfWords = X_train.shape
        self.classes_ = np.unique(Y_train)
        noOfClasses = len(self.classes_)
        
        self.class_prior = np.zeros(noOfClasses)
        for index, cls in enumerate(self.classes_):
            self.class_prior[index] = np.sum(Y_train == cls) / noOfDocs
        
        self.conditional_prob = np.zeros((noOfClasses, noOfWords))
        for index, cls in enumerate(self.classes_):
            xTrain_cls = X_train[Y_train == cls]
            class_word_count = xTrain_cls.sum(axis=0)
            total_class_word_count = class_word_count.sum()
            self.conditional_prob[index, :] = (class_word_count + self.alpha) / (total_class_word_count + noOfWords)  # Laplace smoothing
    
    #function predict to test the trained data     
    def predict(self, X_train):
        noOfDocs = X_train.shape[0]
        probabilityLog = np.zeros((noOfDocs, len(self.classes_)))
        
        for index, cls in enumerate(self.classes_):
            probabilityLogCls = np.log(self.class_prior[index])
            probabilityLogWords = X_train @ np.log(self.conditional_prob[index, :].T)
            probabilityLog[:, index] = probabilityLogCls + probabilityLogWords
        
        return self.classes_[np.argmax(probabilityLog, axis=1)]
    
    #function to evaluate the predicted data using ROC and AUC
    def predict_proba(self, X_train):
        noOfDocs = X_train.shape[0]
        probabilityLog = np.zeros((noOfDocs, len(self.classes_)))
        
        for index, cls in enumerate(self.classes_):
            probabilityLogCls = np.log(self.class_prior[index])
            probabilityLogWords = X_train @ np.log(self.conditional_prob[index, :].T)
            probabilityLog[:, index] = probabilityLogCls + probabilityLogWords
        
        return np.exp(probabilityLog - np.max(probabilityLog, axis=1, keepdims=True))
    
#%%
# a function to perform hypertuning
def hypertune_multinomial_nb(X_train, Y_train, param_grid):
    grid_search = GridSearchCV(MultinomialNaiveBayes(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, Y_train)
    
    best_alpha = grid_search.best_params_['alpha']
    best_model = MultinomialNaiveBayes(alpha=best_alpha)
    best_model.fit(X_train, Y_train)
    
    return best_model, grid_search.best_params_, grid_search.best_score_

#to define the parameter grid
param_grid = {'alpha': [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0]}

# to apply the hypertune function on the train data
best_model, best_params, best_score = hypertune_multinomial_nb(X_train_main, Y_train_main, param_grid)

print("Best Parameters:", best_params)
print("Best CV Score:", best_score)

#%%
# to predict using the best model
predictions = best_model.predict(X_test)
proba_predictions = best_model.predict_proba(X_test)
print("Predictions:", predictions)

#%%
# to evaluate using the accuracy, precision, recall and f1-score metrics
accuracy = accuracy_score(Y_test, predictions)
precision = precision_score(Y_test, predictions, average='macro')
recall = recall_score(Y_test, predictions, average='macro')
f1 = f1_score(Y_test, predictions, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
print(f"F1 Score: {f1}")

#%%
# to get the Confusion Matrix
conf_matrix = confusion_matrix(Y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

#%%
# to print classification report
target_names = [str(cls) for cls in best_model.classes_]
class_report = classification_report(Y_test, predictions, target_names=target_names)
print("\nClassification Report:\n", class_report)

#%%
# to get the ROC and AUC
Y_test_bin = label_binarize(Y_test, classes=best_model.classes_)
n_classes = Y_test_bin.shape[1]

false_positive_rate = dict()
true_positive_rate = dict()
roc_auc = dict()

for i in range(n_classes):
    false_positive_rate[i], true_positive_rate[i], _ = roc_curve(Y_test_bin[:, i], proba_predictions[:, i])
    roc_auc[i] = roc_auc_score(Y_test_bin[:, i], proba_predictions[:, i])

# to plot the ROC for each class
plt.figure()
for i in range(n_classes):
    plt.plot(false_positive_rate[i], true_positive_rate[i], label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {best_model.classes_[i]}')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Multiclass Classification')
plt.legend(loc="lower right")
plt.show()

# %%
