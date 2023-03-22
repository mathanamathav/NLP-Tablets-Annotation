<div align="center">
<h1> NLP-Tablets-Annotation
</h1>

<p>
Given a set of tablets images, do OCR, convert the image to text, extract necessary details such as name of medicine, molecules in it, date of manufacturing, date of expiry. Convert this text into speech. This can be done by creating a drug database by scraping drug details and form a lexicon. Can use api for text to speech conversion
</p>

<p align="center">
   <img src="https://skillicons.dev/icons?i=python,github" />
</p>

![Streamlit](https://img.shields.io/static/v1?style=for-the-badge&message=Streamlit&color=FF4B4B&logo=Streamlit&logoColor=FFFFFF&label=)
<hr>
</div>

![CPT2303191942-800x800](https://user-images.githubusercontent.com/62739618/226181871-0e53c1d8-fd84-429d-a6fb-a6818c0cd0db.gif)

Checkout the report made for this project [here](https://curvy-mimosa-fce.notion.site/NLP-Tablets-Annotation-08d6e264d12b401c847d43e9f844f773).

Download the required model from [here](https://drive.google.com/drive/folders/1wvtpL5KASuDovq_3AdhAgutqQM3waEYO?usp=sharing) add it to model folder.
  
  ## Tasks Done
- [x] Scraping Medicine Images from https://www.netmeds.com/prescriptions
- [x] PreProcessing Image
- [x] Extracting Text from Image using Paddle OCR
- [x] Generating Vocabulary of Words for Spelling Correction
- [x] Spelling Corection using Minimum Edit Distance
- [x] Annotating Text from Training Corpus
- [x] Training NER Model using Spacy with Training data
- [x] Extracting Required Entities from given text
- [x] Forming Lexicon for different categories of Image
- [x] Displaying the eend results in Web App built using Streamlit


## Datasets Used
* https://www.netmeds.com/prescriptions
* https://www.kaggle.com/datasets/shudhanshusingh/az-medicine-dataset-of-india

## Libraries Used
* PaddleOCR - For Text Extraction from Image
* Spacy - For Training NER Model
* CV2 - For Image processing
* NLTK - For Text processing
* TTS - Google API for text to speech conversion
* Streamlit - For building Web Application

## Colab Links
* https://drive.google.com/file/d/1Oefz3h4L8HuvCmqtIBojE8CEd2AelaFi/view?usp=sharing
* https://drive.google.com/file/d/1rHzwlU06lhOCP87o3-0VF8hz5Nu_t3km/view?usp=share_link
* https://drive.google.com/file/d/1YeHwZvFBbcqymMpn0uyOvYScxxL43mns/view?usp=sharing




