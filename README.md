# Sentiment_Analysis

**Sentiment Analysis** is defined as a process of computationally identifying and categorizing opinions from the piece of text and determining whether the writer's attribute towards a particular topic or towards a product is positive, negative or neutral. Sentiment Analysis is also known as Opinion Mining.

The main **purpose** of the project is to detect the hate speech from the extracted tweets. We will categorize the tweet as hate speech if the tweet contains the negative sentiment. In our project, we use a **Logistic Regression algorithm** to build a sentiment analysis model. This algorithm predicts the probability of occurrence of an event by fitting a data to the logit function.

**METHODOLOGY**

Sentiment classification can be done in the following ways: 
1. Text pre-processing and cleaning
   
    a. Removingtwitterhandles
   
    b. Removing punctuations, numbers and special characters
   
    c. Removingshortwords
   
    d. Tokenization
   
    e. Stemming


    <img width="533" alt="Screenshot 2023-06-14 at 6 42 04 PM" src="https://github.com/Jaswanthyadla/Sentiment_Analysis/assets/36241001/80e940c1-eef3-4e96-8091-cd9fc5a4c1bc">

          After Data Cleaning performed, top 5 tweets looks as above.

   
3. Visualisation from tweets

    We used Wordcloud for visualisation of data. In simple words, word clouds are defined as a cluster of words depicted in different sizes. This is an ideal way to pull out the most pertinent parts of textual data, from blog posts to databases.

   
   <img width="300" alt="Screenshot 2023-06-14 at 6 29 16 PM" src="https://github.com/Jaswanthyadla/Sentiment_Analysis/assets/36241001/7ae1e51b-92a4-44cf-896c-41622b7ff483">

   <img height="200" width="480" alt="Screenshot 2023-06-14 at 6 33 06 PM" src="https://github.com/Jaswanthyadla/Sentiment_Analysis/assets/36241001/44d14483-9a3b-43df-810c-4c4ce28fb9d6">
   
5. Feature extraction from cleaned tweets
   
   The mapping of textual data to real valued vectors is called Feature extraction. In our project we have used Bag of Words and TF-IDF methods to extract the necessary features for classification purpose.
   
6. Building the model (training)
   
   The Algorithm that we used is **Logistic Regression** which is a classification algorithm. It is used to predict a binary outcome (1 / 0, Yes / No, True / 
   False) given a set of independent variables.  We use the Sigmoid function/curve to predict the categorical value. The threshold value decides the 
   outcome(win/lose). we used **F1- score** as an evaluation metric for our machine. Accuracy is used when the True Positives and True negatives are more important       while F1-score is used when the False Negatives and False Positives are crucial.
