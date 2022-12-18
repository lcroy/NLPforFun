import streamlit as st

from PIL import Image
from configuration import Config
from main.SU.summarization import *
from main.PM.trmForPoem import *
from main.IG.craiyon.craiyon import Craiyon

import requests

# config file defines the necessary parameters
cfg = Config()

def dis_home_page():
    st.title(cfg.home_title)
    st.markdown(cfg.home_des)

def dis_TG_page():
    st.title(cfg.TG_title)
    st.markdown(cfg.TG_des)
    img = Image.open(cfg.TG_image_path)
    col1, col2, col3 = st.columns([2, 6, 1])
    with col1:
        st.write("")
    with col2:
        st.image(img, width=400)
    with col3:
        st.write("")
    message = st.text_area("Enter your snippet", "Type Here")
    click = st.button("Let's make a story...")
    generator = pipeline('text-generation', model='gpt2')
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


def dis_PM_page():
    st.title(cfg.PM_title)
    st.markdown(cfg.PM_des)
    img = Image.open(cfg.PM_image_path)
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.write("")
    with col2:
        st.image(img, width=450)
    with col3:
        st.write("")
    message = st.text_area("Enter your snippet", "Type Here")
    click = st.button("Let's create a poem...")
    max = st.sidebar.slider('Select max', 50, 500, step=10, value=150)
    max_sequence_len = 12
    if click:
        with st.spinner("Wait..."):
            sentence = gen_poem(message, max_sequence_len, max)
        # st.write(text)
        st.success(sentence)


def dis_IG_page():
    st.title(cfg.IG_title)
    st.markdown(cfg.IG_des)
    img = Image.open(cfg.IG_image_path)
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.write("")
    with col2:
        st.image(img, width=450)
    with col3:
        st.write("")
    message = st.text_area("Please write a short description of the image you want to generate.")
    click = st.button("Let's create an image...")
    if click:
        with st.spinner("Wait..."):
            generator = Craiyon()  # Instantiates the api wrapper
            result = generator.generate(message)
            result.save_images()
    # load images
    st.write('---')
    st.write("last generated image...")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        st.image(Image.open(cfg.IG_dalle_image_path + '/image-1.png'))
    with col2:
        st.image(Image.open(cfg.IG_dalle_image_path + '/image-2.png'))
    with col3:
        st.image(Image.open(cfg.IG_dalle_image_path + '/image-3.png'))
    with col4:
        st.image(Image.open(cfg.IG_dalle_image_path + '/image-4.png'))



def dis_SU_page():
    st.title(cfg.SU_title)
    st.markdown(cfg.SU_des)
    img = Image.open(cfg.SU_image_path)


    st.image(img, width=700)

    summarizer = load_summarizer()

    message = st.text_area("Please enter your text here", "Type Here")
    click = st.button("Let's summarize it...")

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
    # change the navigation
    if nav_choice == 'Home':
        dis_home_page()
    elif nav_choice == 'AI Applications':
        NLP_choice = st.sidebar.selectbox("Select Activity", cfg.NLP_menu)
        if NLP_choice == 'Make a story':
            dis_TG_page()

        elif NLP_choice == 'Discovering':
            dis_SU_page()

        elif NLP_choice == 'Poetry generation':
            dis_PM_page()

        elif NLP_choice == 'Turn language into Art':
            dis_IG_page()

    # elif nav_choice == 'Computer Vision':
    #     CV_choice = st.sidebar.selectbox("Select Activity", cfg.CV_menu)
    #     if CV_choice == 'DeepFake':
    #         dis_home_page()
    #
    #     elif CV_choice == 'Other':
    #         dis_home_page()

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
