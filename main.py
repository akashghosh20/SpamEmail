import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
with open("EmailSpamChecker.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the fitted TF-IDF vectorizer
with open("TFIDFVectorizer.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

st.title("Email Spam Checker")

# Input field for the user to enter an email
user_input = st.text_area("Enter the email text:")

if st.button("Check Spam"):
    if not user_input:
        st.warning("Please enter an email.")
    else:
        # Transform the user's input using the loaded vectorizer
        input_data_features = tfidf_vectorizer.transform([user_input])

        # Make a prediction
        prediction = model.predict(input_data_features)

        # Display the prediction result
        if prediction[0] == 0:
            st.error("This email is classified as spam.")
        else:
            st.success("This email is classified as ham (not spam).")

# Add an optional section for displaying more information about the model or the problem.
st.sidebar.markdown("## About")
st.sidebar.info(
    "This is a simple web app that uses a trained model to classify emails as spam or ham."
)
