import streamlit as st
import PyPDF2
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk import sent_tokenize
from string import punctuation
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.sidebar.header('PDF Reader Chatbot')

pdf_file = st.sidebar.file_uploader("Upload a text PDF file only",type ='pdf')


if pdf_file is not None:
    pdfReader = PyPDF2.PdfReader(pdf_file)
    
    st.write('Total no of Pages in pdf',len(pdfReader.pages), 'and please upload text pdf file only')
    text = str()
    for i in range(4,20): 
        pageObj = pdfReader.pages[i]
        page = pageObj.extract_text()
        page = re.sub(r'[^\x20-\x7E]', ' ', page)
        page = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', page)
        page = re.sub(r'\n',' ', page)
        text = text + page

    text=text.lower()
    
    sent_token = sent_tokenize(text)
    
    punc = punctuation
    def remove_punc(text):
        for char in punc:
            text=text.replace(char,' ')
        return text
    
    def word_token(text):
        return word_tokenize(text)    
    
    lemma = WordNetLemmatizer()
    def lemmatizer(words):
        return[lemma.lemmatize(word) for word in words]
    
    stop_words = stopwords.words('english')
    def remove_stopword(words):
        return [word for word in words if not word in stop_words]
    
    
    word_token_sent=[]
    for sent in sent_token:
        word_token_sent.append(lemmatizer(remove_stopword(word_token(remove_punc(sent)))))
    
    sentence=[" ".join(li) for li in word_token_sent]
    
    word_token_text=[]
    for sent in sent_token:
        word_token_text.extend(lemmatizer(remove_stopword(word_token(remove_punc(sent)))))
    
    def responce(user_input):
        bot_response=''
        tfidfvec= TfidfVectorizer()
        tfidf = tfidfvec.fit_transform(sentence)
        vals = cosine_similarity(tfidf[-1],tfidf)
        idx= vals.argsort()[0][-2]
        flat= vals.flatten()
        flat.sort()
        req_tfidf=flat[-2]
        if req_tfidf==0:
            return 'no words for that query'
        else:
            bot_responce =  sent_token[idx]
            return bot_responce 

    flag= True
    while(flag==True):
        user_input=st.text_input("Enter your Query here. If no Query or exit type 'bye' and don't bother about error just write your query",)
        user_input=user_input.lower()
        
        if user_input =='bye':
            flag=False
        elif user_input !=None:
            user_input=" ".join(lemmatizer(remove_stopword(word_token(remove_punc(user_input)))))
            sentence.append(user_input)
            word_token_text = word_token_text + word_token(user_input)
            final_words = list(set(word_token_text))
            st.sidebar.write(responce(user_input))
            sentence.remove(user_input)


