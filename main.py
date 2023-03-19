import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import spacy
from gtts import gTTS
from paddleocr import PaddleOCR
import cv2
import numpy as np
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


@st.cache_resource
def create_ocr():
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
    return ocr


@st.cache_resource
def get_model():
    filepath = r"model\spacy_model.pkl"
    nlp = pd.read_pickle(filepath)
    return nlp


@st.cache_resource
def word_chart(text):
    """
    """
    word_list = text.split()
    word_frequency = {}
    for word in word_list:
        if word in word_frequency:
            word_frequency[word] += 1
        else:
            word_frequency[word] = 1

    word_df = pd.DataFrame.from_dict(
        word_frequency, orient='index', columns=['frequency'])
    wordcloud = px.scatter(word_df, x=word_df.index,
                           y='frequency', text='frequency')

    return wordcloud


@st.cache_resource
def word_cloud(text):
    wordcloud = WordCloud().generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


def img_processing(image):
    """
    """
    img = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    dst = img[y:y+h, x:x+w]
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    dst = cv2.dilate(dst, kernel, iterations=1)
    dst = cv2.erode(dst, kernel, iterations=1)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    return dst


def takeText(preprocessed_image, ocr):
    """
    """
    result = ocr.ocr(preprocessed_image, cls=True)[0]
    boxes = [res[0] for res in result]
    texts = [res[1][0] for res in result]
    scores = [res[1][1] for res in result]
    cols = []
    for x in result:
        yes = True
        if cols == []:
            cols.append(x[0][0][0])
        else:
            for i in cols:
                if abs(x[0][0][0] - i) <= 10:
                    yes = False
                    break
            if yes:
                cols.append(x[0][0][0])
    txts = []
    for x in cols:
        for res in result:
            if abs(res[0][0][0] - x) < 10:
                txts.append(res[1][0])

    return ' '.join(txts)


def tts(text):
    tts = gTTS(text=text, lang='en')
    tts.save("tts.mp3")
    # return ipd.Audio("tts.mp3")
    return 'tts.mp3'


def annotate():
    st.snow()
    nlp = get_model()
    ocr = create_ocr()
    uploaded_file = st.file_uploader("Choose a JPEG Image file", type='jpg')

    st.text(" ")
    if uploaded_file is not None:
        with st.spinner('Annotating image ...'):

            file_bytes = np.asarray(
                bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            preprocessed_image = img_processing(image)
            ocr_text = takeText(preprocessed_image, ocr)
            preprocessed_image = cv2.resize(
                preprocessed_image, dsize=(600, 600))
            # preprocessed_image = preprocessed_image.resize((600,600))
            doc = nlp(ocr_text)

        st.text(" ")
        col1, col2 = st.columns(2)
        with col1:
            st.header("Input Image")
            st.image(image)

        with col2:
            st.header("Preprocessed Image")
            st.image(preprocessed_image)
        st.text(" ")

        st.success("✅ Annotation Done!")

        colors = {"COMPONENT": "#F67DE3",
                  "NAME": "#7DF6D9", "MANUFACTURER": "#FFFFFF"}
        options = {"colors": colors}
        html = spacy.displacy.render(doc, style="ent", options=options)
        st.write(html, unsafe_allow_html=True)
        annotated_COMPONENT = []
        annotated_NAME = []
        annotated_MANUFACTURER = []
        text = ""

        for ent in doc.ents:
            if ent.label_ == 'COMPONENT':
                annotated_COMPONENT.append(ent.text)
            elif ent.label_ == 'NAME':
                annotated_NAME.append(ent.text)
            elif ent.label_ == 'MANUFACTURER':
                annotated_MANUFACTURER.append(ent.text)

        if annotated_NAME is not None:
            text += "The Given Medicine is"+" ".join(annotated_NAME)
        if annotated_COMPONENT is not None:
            text += " which consists of "+" ".join(annotated_COMPONENT)
        if annotated_COMPONENT is not None:
            text += " and is manufactured by "+" ".join(annotated_MANUFACTURER)

        st.text(" ")
        col1, col2, col3 = st.columns(3)
        if col2.button('Speak'):
            file = tts(text)
            audio = open(file, 'rb').read()
            st.audio(audio, format='audio/mp3')

        st.text(" ")
        st.header('Word Frequency Count chart')
        st.plotly_chart(word_chart(ocr_text),
                        use_container_width=True, theme="streamlit")

        st.text(" ")
        st.header('Word Cloud chart')
        word_cloud(ocr_text)


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


def lexicon():

    option = st.selectbox(
        'Select Category of Disease',
        ('ADHD', 'Digestion', 'Heart', 'Liver'))
    if option == 'ADHD':
        st.header("LEXICON FOR ADHD MEDICINES")
        df = pd.read_csv(r"F:\Downloads\ADHD_annotated.csv")
        st.dataframe(df.head(10))
        csv = convert_df(df)
        st.download_button(
            "Press to Download Lexicon",
            csv,
            "file.csv",
            "text/csv",
            key='download-csv'
        )

    if option == 'Digestion':
        st.header("LEXICON FOR DIGESTION MEDICINES")
        df = pd.read_csv(r"F:\Downloads\digestion-annotated.csv")
        st.dataframe(df.head(10))
        csv = convert_df(df)
        st.download_button(
            "Press to Download Lexicon",
            csv,
            "file.csv",
            "text/csv",
            key='download-csv'
        )

    if option == 'Heart':
        st.header("LEXICON FOR HEART MEDICINES")
        df = pd.read_csv(r"F:\Downloads\Heart_annotated.csv")
        st.dataframe(df.head(10))
        csv = convert_df(df)
        st.download_button(
            "Press to Download Lexicon",
            csv,
            "file.csv",
            "text/csv",
            key='download-csv'
        )

    if option == 'Liver':
        st.header("LEXICON FOR LIVER MEDICINES")
        df = pd.read_csv(r"F:\Downloads\liver-disease-annotated.csv")
        st.dataframe(df.head(10))
        csv = convert_df(df)
        st.download_button(
            "Press to Download Lexicon",
            csv,
            "file.csv",
            "text/csv",
            key='download-csv'
        )


def main():
    st.title("Medical Image Annotator")

    menu = ["Annotation", "Lexicon"]
    choice = st.sidebar.selectbox("Metrics", menu)

    if choice == "Annotation":
        annotate()
    if choice == 'Lexicon':
        lexicon()


if __name__ == '__main__':
    main()
