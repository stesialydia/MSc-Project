#%%
#import the libraries for the performing and categorizing sentiments 
import spacy
import pandas as pd
import preprocessScript
import categorizingSentimentScript
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

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