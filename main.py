import pyttsx3
from transformers import pipeline
import speech_recognition as sr
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Initialize text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Typically female voice, adjust as needed
engine.setProperty('volume', 10.0)
engine.setProperty('rate', engine.getProperty('rate') - 25)

def speak(engine, text):
    engine.say(text)
    engine.runAndWait()

def sentiment_analysis_pipeline():
    # Use sentiment-analysis pipeline from transformers
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", tokenizer="distilbert-base-uncased-finetuned-sst-2-english")

def emotional_support_pipeline(text):
    analyzer = sentiment_analysis_pipeline()
    result = analyzer(text)[0]
    sentiment = result['label']

    if sentiment == "POSITIVE":
        return "That's wonderful to hear! Do you want to share more?"
    elif sentiment == "NEGATIVE":
        return "I'm sorry you're feeling this way. Remember, it's okay to feel your emotions. Let's talk about it."
    else:
        return "I'm here for you. Let's chat!"

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio_data = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            speak(engine, "Sorry, I couldn't understand that.")
            return None

def main():
    speak(engine, "Hello! How can I help you today?")
    while True:
        speak(engine, "How are you feeling today?")
        text = get_voice_input()
        if text:
            response = emotional_support_pipeline(text)
            print(f"You said: {text}")
            speak(engine, response)

if __name__ == "__main__":
    main()