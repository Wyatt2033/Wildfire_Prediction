import  streamlit as st
import pandas as pd

def upload_file(file):
    uploaded_file = st.file_uploader(file, type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)
        return df


def write(param):
    return None