import streamlit as st
import pandas as pd #turning data into table
from textblob import TextBlob# python lib for sentiment analysis
import cleantext
import altair as alt#for visulaisation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import plotly.express as px
import joblib
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
#from nltk.tokenize import tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd
#Model=joblib.load(open("models/customer_review.pkl","rb"))

# Load model
model = tf.keras.models.load_model("models/modelReview.h5")


#fonction to get comments
def get_comment(a):
    dataset = pd.read_csv("data.csv")
    padding_type='post'
    vocab_size = 1000
    max_length = 100
    oov_tok = "<OOV>"
    l=[]
    l.append(a) 
    sentences = dataset['text'].tolist()
    labels = dataset['sentiment'].tolist()
    # Separate out the sentences and labels into training and test sets
    training_size = int(len(sentences) * 0.8)
    training_sentences = sentences[0:training_size]
    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)    
    sample_sequences = tokenizer.texts_to_sequences(l)
    fakes_padded = pad_sequences(sample_sequences, padding=padding_type, maxlen=max_length)           
    classes = model.predict(fakes_padded)
    return classes








st.title(' Sentiment Analysis App')#create   title
st.markdown("this application is all about analysis of the impression of customers on a product based on their comments.")
st.subheader("feel free and express your point of view")

st.sidebar.title('Customers impression')
st.image("https://lasagadesaudacieux.com/wp-content/uploads/2021/07/produit.jpg")
data=pd.read_csv('data.csv')



with st.expander('Add your Comment'):
    text=st.text_input('tap here :')
    if text:#if there is an input text
        blob=TextBlob(text)
  
    if text:
      
        st.write("clean comment   : " , cleantext.clean(text,clean_all=False,extra_spaces=True,
        stopwords=True,lowercase=True,numbers=True,punct=True))

   

def convert_to_df(sentiment):
    sentiment_dict={"polarity":sentiment.polarity,"subjectivity":sentiment.subjectivity}
    sentiment_df=pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
    return sentiment_df      

def analyze_token_sentiment(docx):
    analyzer=SentimentIntensityAnalyzer()
    postive=[]
    negative=[]
    neutral=[]
    for i in docx.split():
        res=analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            postive.append(i)
            postive.append(res)  
        elif res<= -0.1:
            negative.append(i)
            negative.append(res)
        else:
            neutral.append(i)
            neutral.append(res)
    result={'postive':postive,'negative':negative,'neutral': neutral}
    return result

if st.button('Predict'):
    col1, col2 = st.columns(2)   
    classes=get_comment(text) 

    with col1:
        st.info("Results")
        st.success("Original comment")
        st.write(text)
        st.success("Prediction :")
        st.write(classes)
        st.success("This comment is :")
        sentiment=TextBlob(text).sentiment
        #st.write(sentiment)#afficher Polarity and subjectivity
        if sentiment.polarity >0:#positive
            st.markdown("Positive :smiley: ")
        elif sentiment.polarity <0:
            st.markdown(" Negative :angry: ")
        else:
            st.markdown(": Neutral 😐 ")
        
        #dataframe
        result_df=convert_to_df(sentiment)
        st.dataframe(result_df)

        #visualisation
        c=alt.Chart(result_df).mark_bar().encode(
            x='metric',
            y='value',
            color='metric'
        )
        st.altair_chart(c,use_container_width=True)
        st.write("La polarité est une valeur flottante dans la plage [-1,0 à 1,0] où 0 indique neutre, +1 indique un sentiment très positif et -1 représente un sentiment très négatif.")
        
        st.write("La subjectivité est une valeur flottante dans la plage [0,0 à 1,0] où 0,0 est très objectif et 1,0 est très subjectif.")



    with col2:
        st.info("Word sentiment")
        token_sentiment=analyze_token_sentiment(blob)
        st.write(token_sentiment)
       









def Score(x):
    blob1=TextBlob(x)

    return blob1.sentiment.polarity


def analyze(x):
    if x>=0.5:
        return 'Positiive'
    elif x<= -0.5:
        return 'Negative'
    else:
        return 'Neutral'




if st.checkbox("show data"):
    data['score']=data['text'].apply(Score)
    
    data['analysis']=data['sentiment'].apply(analyze)
    st.write(data.head(10))

    select=st.sidebar.selectbox("visualisation of data",['Histogram','Pie chart'],key=1)
    sentiment=data['sentiment'].value_counts()
    sentiment=pd.DataFrame({ 'sentiment':sentiment.index,'comments':sentiment.values})
    st.markdown('#### Sentiment count')
    if select=='Histogram':
        fig=px.bar(sentiment,x="sentiment",y='comments',color='comments',height=500)
        st.plotly_chart(fig)
    else:
        fig=px.pie(sentiment,values="comments",names="sentiment")
        st.plotly_chart(fig)










