from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re
import pickle
import streamlit as st
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')



st.title("Spam Classifier🚫📧")


sentence =st.text_input("Enter Message")

# cleaning and preprocessing

sentence =re.sub("[^a-zA-Z]", " ", sentence)
sentence =sentence.lower().split()
sentence =[WordNetLemmatizer().lemmatize(word) for word in sentence  if word not in stopwords.words("english")]
sentence =" ".join(sentence)


# importing tf-idf, model
with open("model.pkl", "rb") as f:
    model =pickle.load(f)

with open("tf_idf_object.pkl", "rb") as f:
    tf_idf =pickle.load(f)


# transoforming input
sentence =tf_idf.transform([sentence]).toarray()

# prediction
if model.predict(sentence)==1:
    st.write("Spam")
else:
    st.write("Ham")



    

