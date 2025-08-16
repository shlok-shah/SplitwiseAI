import streamlit as st
import pandas as pd
import os
from PIL import Image
from dotenv import load_dotenv
from bill_process import get_bill_details, get_dataframes_using_convo
from workingsplit import *

# Google Gemini
import google.generativeai as genai

# Streamlit Setup
st.set_page_config(layout="wide")
load_dotenv()

# Initialize Gemini client
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

uploaded_image = st.file_uploader("Upload Your Bill Image", type=["png", "jpg", "jpeg"])

members = get_members()
member_names = [member.first_name for member in members]

if uploaded_image is not None:
    with open("download.jpg", "wb") as file:
        file.write(uploaded_image.getbuffer())
    image = Image.open(uploaded_image)

    paidby = st.sidebar.radio("Who paid the bill?", member_names)
    idpaidby = members[member_names.index(paidby)].id
    st.sidebar.image(image, caption='Uploaded Image.', width=300)

    if "df" not in st.session_state:
        with st.spinner("Extracting and analyzing bill..."):
            bill_details = get_bill_details(model)
            df = get_dataframes_using_convo(model, bill_details)
            st.session_state["df"] = df
            df.to_csv('bill.csv', index=False)
    else:
        df = st.session_state["df"]

    st.sidebar.write(df)

    # UI for selecting members per item
    for item in range(len(df)):
        item_name = df.iloc[item]['Item Name']
        cols = st.columns(len(members) + 1)
        for i in range(len(cols)):
            if i == 0:
                cols[i].write(item_name)
            else:
                checkbox_key = f"{item}_{members[i - 1].id}"
                cols[i].checkbox(members[i - 1].first_name, key=checkbox_key)
        st.divider()

    # Button to confirm split
    if st.button("Create Split"):
        st.write("Create Split clicked!")
        mat = [[] for _ in range(len(df))]

        # Collect selected checkboxes
        for k in st.session_state.keys():
            if "_" in k and st.session_state[k] == True:
                try:
                    item, user = map(int, k.split("_"))
                    mat[item].append(user)
                except ValueError:
                    continue

        # Perform split
        for i in range(len(mat)):
            if len(mat[i]) > 1:
                item_name = df['Item Name'][i]
                total_amount = df['Total'][i]
                expense = create_split(idpaidby, mat[i], total_amount, item_name)
                if expense[1] is not None:
                    st.error(f"Error in item '{item_name}': {expense[1].errors}")
                else:
                    st.success(f"✅ Split created for item: {item_name}")
