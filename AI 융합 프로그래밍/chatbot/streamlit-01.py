import streamlit as st
import numpy as np
import pandas as pd
import openai

openai.api_key = ""
openai.api_version = "2023-05-15"
openai.api_type = "azure"
openai.azure_endpoint = "https://sktflyai.openai.azure.com/"

st.title("welcome to GPT ^^")

subject = st.text_input("시의 제목을 입력하세요")
content = st.text_area("시의 내용을 입력하세요")

button_click = st.button("확인")


if button_click:
    with st.spinner("wait for it..."):
        result = openai.chat.completions.create(
            model="dev-gpt-35-turbo",
            temperature=1,  # 1까지만 쓰는걸로
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },  # 우리는 그냥 쓰지만 뒷단에서 gpt에 전부 설정되어 있음
                {"role": "user", "content": f"시의 주제는 {subject}"},
                {"role": "user", "content": f"시의 내용은 {content}"},
                {"role": "user", "content": "이 내용으로 시를 써줘"},
            ],
        )
        st.success("Done!")
    st.write(result.choices[0].message.content)
