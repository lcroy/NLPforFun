"""
## NLP Application

DESCRIPTION

Author: [Chen Li](https://vbn.aau.dk/en/persons/142294)
Source: [Github](https://github.com/TBC)
"""
import streamlit as st

from PIL import Image
from numpy.random import seed
from configuration import Config
from main.SA.sen_twitter import TwitterClient
from main.QA.bert import QA
from main.SU.summarization import *

# config file defines the necessary parameters
cfg = Config()

def dis_home_page():
    st.title(cfg.home_title)
    st.markdown(cfg.home_des)

def dis_QA_page():
    st.title(cfg.QA_title)
    st.markdown(cfg.QA_des)
    img = Image.open(cfg.QA_image_path)
    st.image(img, width=700)
    st.write('---')
    message_context = st.text_area("Give some context first", "Type Here")
    message_question = st.text_area("You may ask me a question now", "Type Here")
    click = st.button("Show me answer")
    if click:
        with st.spinner("Wait..."):
            qa = QA(cfg.QA_model_path)
            answer = qa.predict(message_context.title(), message_question.title())
        st.success(answer['answer'])
    # model details
    st.write('---')
    st.header("Model Details")
    st.subheader("Information-retrieval based Qeustion Answer")
    st.markdown(cfg.QA_process)
    st.subheader("Model Architecture")
    st.markdown(cfg.QA_model_overview)
    img = Image.open(cfg.QA_bert_model_image_path)
    st.image(img, width=550)
    st.write('---')
    st.header("Reference")
    st.info(cfg.QA_about)


def dis_TG_page():
    st.title(cfg.TG_title)
    st.markdown(cfg.TG_des)
    img = Image.open(cfg.TG_image_path)
    st.image(img, width=200)
    message = st.text_area("Enter your snippet", "Type Here")
    click = st.button("Generate Response")
    generator = pipeline('text-generation', model='gpt2')
    seed(42)
    if click:
        with st.spinner("Wait..."):
            sentence = generator(str(message.title()), max_length=100, num_return_sequences=1)
        st.success(sentence[0]['generated_text'])
    # model details
    st.write('---')
    st.header("Model Details")
    st.subheader("What is GPT2")
    st.markdown(cfg.TG_model_overview)
    st.subheader("Model Architecture")
    img = Image.open(cfg.TG_architecture_image_path)
    st.image(img, width=550)

    st.write('---')
    st.header("About")
    st.info(cfg.TG_about)


def dis_SA_page():
    st.title(cfg.SA_title)
    st.markdown(cfg.SA_des)
    img = Image.open(cfg.SA_image_path)
    st.image(img, width=700)
    st.header("Analysis Twitter Topic.")
    message = st.text_area("Let's play with tweet first","Type Here")
    click = st.button("Sentiment Detection for tweet")
    if click:
        with st.spinner("Wait..."):
            sa = TwitterClient()
            ptweetsPer, ntweetsPer, netweetsPer, ptweets, ntweets, netweets = sa.run(message.title())
            # percentage results
            per_results = ptweetsPer + '\n\r' + ntweetsPer + '\n\r' + netweetsPer
            # details results
            pt = [item['text'] for item in ptweets]
            nt = [item['text'] for item in ntweets]
            net = [item['text'] for item in netweets]

        st.success("Result Distribution: \n\r" + per_results)
        st.text("Positive Tweets:")
        st.info(pt)
        st.text("Negative Tweets: ")
        st.info(nt)
        st.text("Neutral Tweets:")
        st.info(net)
    st.write('---')
    st.header("Analysis sentences.")
    sentence = st.text_area("Let's give a sentence", "Type Here")
    click_sen = st.button("Sentiment Detection for sentence")
    if click_sen:
        with st.spinner("Wait..."):
            classifier = pipeline("sentiment-analysis")
            results = classifier([sentence.title()])

        st.success("The sentence is predicted as a " + str(results[0]['label']) + " and the score is " + str(results[0]['score']))


def dis_SU_page():
    st.title(cfg.SU_title)
    st.markdown(cfg.SU_des)
    img = Image.open(cfg.SU_image_path)
    st.image(img, width=500)

    summarizer = load_summarizer()

    message = st.text_area("Please enter your text here", "Type Here")
    click = st.button("Generate Summarization")

    max = st.sidebar.slider('Select max', 50, 500, step=10, value=150)
    min = st.sidebar.slider('Select min', 10, 450, step=10, value=50)
    do_sample = st.sidebar.checkbox("Do sample", value=False)
    if click and message:
        with st.spinner("Generating Summary.."):
            chunks = generate_chunks(message)
            res = summarizer(chunks,
                             max_length=max,
                             min_length=min,
                             do_sample=do_sample)
            sentence = ' '.join([summ['summary_text'] for summ in res])
        # st.write(text)
        st.success(sentence)
    st.write('---')
    st.header("Model Details")
    st.subheader("What is BART")
    st.markdown(cfg.SU_model_overview)
    st.subheader("Model Architecture")
    img = Image.open(cfg.SU_architecture_image_path)
    st.image(img, width=550)

    st.write('---')
    st.header("About")
    st.info(cfg.SU_about)


def set_sidebar():
    # set the navigation menu
    st.sidebar.header('Navigation')
    nav_choice = st.sidebar.radio('', cfg.nav_menu)
    # NLP_choice = st.sidebar.selectbox("Select Activity", cfg.NLP_menu)
    # CV_choice = st.sidebar.selectbox("Select Activity", cfg.CV_menu)
    # RL_choice = st.sidebar.selectbox("Select Activity", cfg.RL_menu)
    # change the navigation
    if nav_choice == 'Home':
        dis_home_page()
    elif nav_choice == 'Natural Language Processing':
        NLP_choice = st.sidebar.selectbox("Select Activity", cfg.NLP_menu)
        if NLP_choice == 'Text Generation':
            dis_TG_page()

        elif NLP_choice == 'Sentiment Analysis':
            dis_SA_page()

        elif NLP_choice == 'Question & Answering':
            dis_QA_page()

        elif NLP_choice == 'Summarization':
            dis_SU_page()

    elif nav_choice == 'Computer Vision':
        CV_choice = st.sidebar.selectbox("Select Activity", cfg.CV_menu)
        if CV_choice == 'Facial Recognition':
            dis_home_page()

        elif CV_choice == 'Object Detection':
            dis_home_page()

        elif CV_choice == 'Other':
            dis_home_page()

    st.sidebar.header('Contributes')
    st.sidebar.info(cfg.contr_info)
    st.sidebar.header('About')
    st.sidebar.info(cfg.abt_info)


def main():
    # import the css style
    with open(r"style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    # set sidebar
    set_sidebar()


if __name__ == "__main__":
    main()
