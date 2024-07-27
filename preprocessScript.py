#%%
#import the libraries for the preprocessing
import spacy
import spacy_cleaner
import pandas as pd

#%%# 
# Load spaCy model
nlp = spacy.load('en_core_web_sm')

#%%
# Load the patient's drug feedback data
df_train_data = pd.read_csv("C:/Users/andre/Downloads/drugsComTrain_raw.csv")
df_test_data = pd.read_csv("C:/Users/andre/Downloads/drugsComTest_raw.csv")

# %%
#to describe the dataset
df_test_data.describe()

#%%
#to display the dataset's features with 10 records
df_test_data.head(10)

# %%
#To combine the train and test dataset 
merge = [df_train_data,df_test_data]
df_review_data= pd.concat(merge,ignore_index=True)

df_review_data.shape   #check the shape of merged_data

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
def preprocess_text(review):
    doc = nlp(review.lower())
    return ' '.join(token.text for token in doc if not token.is_stop and not token.is_punct)

#%%
#Applying the function on the review feature
df_review_data['cleaned_feedback'] = df_review_data['review'].apply(preprocess_text)

#%%
#To check if the cleaned feadback has been added to the dataset as a new feature
df_review_data.head(10)