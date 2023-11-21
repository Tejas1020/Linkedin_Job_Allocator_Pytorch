import streamlit as st
import sqlite3
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import requests
from streamlit_lottie import st_lottie

# Connect to the MySQL database
db_connection = sqlite3.connect(
   'storage.db'
)
db_cursor = db_connection.cursor()

# Create a table for storing user input and model predictions
create_table_query = """
CREATE TABLE IF NOT EXISTS user_input_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_input TEXT,
    model_prediction FLOAT
)
"""
db_cursor.execute(create_table_query)

# Load your PyTorch model
def load_model():
    model_path ="linkedin_job_allocator_bart.pt"
    model = torch.load(model_path, map_location=torch.device('cpu'))  # Adjust map_location based on your needs
    model.eval()
    return model

custom_css = """
    <style>
        .stApp {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #000000; /* Black background */
                color: #FF69B4; /* Pink text color */
            }
    </style>
"""

# Display the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Load the model
model1 = load_model()
# model1 ="C:\\Users\\dsuya\\Desktop\\ \\college\\linkedin_job_allocator_bart.pt"
torch.save(model1.state_dict(), "linkedin_job_allocator_bart.pth")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to("cpu")

model.load_state_dict(torch.load("linkedin_job_allocator_bart.pth", map_location=torch.device('cpu')))
col1,col2=st.columns([2,1], gap="medium")
with col1:
    st.title('Job Allocator')
# Input for user to enter text
    user_input = st.text_area('Enter text for prediction:', '', key='user_input', height = 400)

with col2:
    response = requests.get("https://lottie.host/c4108057-6e4e-46d6-8803-a30747aedf10/qscXeTEX2F.json")
    if response.status_code == 200:
        lottie_data = response.json()
    else:
        st.warning("Failed to fetch Lottie animation data.")
    st_lottie(lottie_data, speed=1, width=300, height=300)
# Pre-Process:
# Load the tokenizer associated with your model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")


tokens = tokenizer(user_input,
                       truncation=True,
                       max_length=256,
                       padding=True,
                       return_tensors="pt").to("cpu")


if st.button('Predict'):
    # st.write(final_input)
    if user_input:
    

        # Display the model's prediction
        output = model.generate(**tokens, max_length=20)
        final_output=tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        with col1:
            st.write('Model Prediction:', final_output)

        # Save user input and model prediction to the database
        insert_query = "INSERT INTO user_input_predictions (user_input, model_prediction) VALUES (?, ?)"
        values = (user_input, final_output)
        db_cursor.execute(insert_query, values)
        db_connection.commit()

# Close the database connection when the app is done
db_cursor.close()
db_connection.close()
