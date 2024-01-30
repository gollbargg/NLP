import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
from audio_recorder_streamlit import audio_recorder

# Initialize the stopwords and PorterStemmer objects outside the function
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def audio_to_text(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data)
    return text


def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    
    # Remove all special characters and digits
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'^b\s+', '', text)
    text = re.sub(r'\d', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
     # Remove Emojis
    text = re.sub(u'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+', '', text)
    
    # Split text into words (tokenization)
    text = text.split()

    # Remove stopwords and perform stemming
    text = [ps.stem(word) for word in text if word not in stop_words]
    
    # Join words to get the text back and return it
    return ' '.join(text)

# Load the trained SVM model and TF-IDF vectorizer
svm_model = joblib.load('svm_classifier2.pkl')
vectorizer = joblib.load('vectorizer2.pkl')

# Define the main function for the Streamlit app
def main():
    st.title("Depression Detector")

    # Choose the input type
    input_type = st.radio("Choose input type", ["Text", "Voice"])

    if input_type == "Text":
        message = st.text_area("Enter Text", "Type Here ..")
    else:
        audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
        if audio_file is not None:
            if audio_file.type == "audio/mp3":
                audio = AudioSegment.from_mp3(audio_file)
                buffer = BytesIO()
                audio.export(buffer, format="wav")
                message = audio_to_text(buffer)
                st.write(f"Transcribed Text: {message}")
            else:
                message = audio_to_text(audio_file)
                st.write(f"Transcribed Text: {message}")
        else:
            message = ""

    if st.button("Analyze"):
        message = clean_text(message)
        data = [message]
        vect = vectorizer.transform(data).toarray()
        my_prediction = svm_model.predict(vect)

        st.success('This text is {}'.format('Depressive' if my_prediction == 1 else 'Not Depressive'))


# Run the main function
if __name__ == '__main__':
    main()
