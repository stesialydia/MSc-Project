#%%
#import the libraries for the performing and categorizing sentiments 
import spacy
import pandas as pd
import preprocessScript
from textblob import TextBlob

#%%
#Performing of sentiment analysis using Textblob to categorize the sentiments
# function to apply sentiment analysis with TextBlob
def analyze_sentiment(cleaned_feedback):
    blob = TextBlob(cleaned_feedback)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

#%%
#Applying the function on the patient's cleaned review
df_review_data[['polarity', 'subjectivity_score']] = df_review_data['review'].apply(lambda x: pd.Series(analyze_sentiment(x)))

# %%
#To test and check if the polarity and subjectivity score features has been added to the dataset.
df_review_data.head(10)

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
#To apply the function for categorising the polarity 
df_review_data['sentiment'] = df_review_data['polarity'].apply(categorize_sentiment)

#%%
#To test and check the added new feature on the dataset
df_review_data.head()
