import streamlit as st
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

# Title of the app
st.title("Text Summarization")

# Text input from the user
text = st.text_area("Summarize your text here.")

# Maximum and Minimum length inputs
max_len = st.text_input("Maximum number of words")
min_len = st.text_input("Minimum number of words")

# Model name for the T5 model
model_name = 't5-large'

# Load the pre-trained T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = TFT5ForConditionalGeneration.from_pretrained(model_name)

def summarize_text(text, max_length, min_length):
    # Tokenize the input text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="tf", max_length=1024, truncation=True)
    # Generate the summary
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def split_text(text, max_chunk_size=1024):
    # Split text into chunks of max_chunk_size
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_length += len(word) + 1  # +1 for the space
        if current_length > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def summarize_long_text(text):
    chunks = split_text(text)
    summaries = [summarize_text(chunk, int(max_len), int(min_len)) for chunk in chunks]
    return " ".join(summaries)

# Button to perform summarization
button = st.button("Summarize")

# Perform summarization on the long text and display the summary
if button:
    if text and max_len.isdigit() and min_len.isdigit():
        summary = summarize_long_text(text)
        st.write(summary)
    else:
        st.error("Please enter valid inputs for text, maximum number of words, and minimum number of words!")