import streamlit as st
from model_app import getCorrectAnswerUsingT5

st.set_page_config(page_title = "Grammar Error correction", page_icon = ":tada:", layout = "wide")

with st.container():
 st.markdown("<h2 style='text-align: center; color: black;'>Grammar Error Correction</h2>", unsafe_allow_html=True)
 st.write("<h4 style='text-align: center; color: black;'>Srinivas Lakshman</h4>", unsafe_allow_html=True)
 st.write("<h5 style='text-align: center; color: black;'>2024SP_MS_DSP_453-DL_SEC61: Natural Language Processing</h5>", unsafe_allow_html=True)
 st.write("<h5 style='text-align: center; color: black;'>Northwestern University</h5>", unsafe_allow_html=True)
 st.write("<h5 style='text-align: center; color: black;'>Dr. Nethra Sambamoorthi | Sudha B G</h5>", unsafe_allow_html=True)


with st.container():
 st.write("---")
 st.write("<h5 style='color: black;'>This is project on crrection of English Grammar. If a sentence is written, the novel model will automaically correct the english grammer. <br>Write an incorrect grammatical sentence and the trained model will correct the sentence.</h5>", unsafe_allow_html=True)


column1, column2 = st.columns(2)

with column1:
    input_incorrect_sentence = st.text_input("Input Incorrect sentence:", key = "input_incorrect_sentence")

with column2:
    input_correct_sentence = st.text_input("Input it's correct sentence:", key = "input_correct_sentence")


if st.button("Get models' generated correct sentence and BLEU score") and input_incorrect_sentence and input_correct_sentence:
 t5_answer, t5_bleu_score = getCorrectAnswerUsingT5(input_incorrect_sentence, input_correct_sentence)
 st.write("T5 model generated answer: ", t5_answer)
 st.write("BLEU score:", t5_bleu_score)
