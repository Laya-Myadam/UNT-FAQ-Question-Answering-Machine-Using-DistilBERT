import pandas as pd
import re
import spacy
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    # Loading spaCy for lemmatization and performing NLP Pre-processing steps
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # Loading the DistilBERT tokenizer and model
    tokenizer_distilbert = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    distilbert_model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')

    # Loading the University FAQ dataset
    faq_data = pd.read_csv('faq_dataset.csv')

    # Additional preprocessing steps with spaCy such as lemmatiation, removing non-relevant characters, converting to lowercase, etc.
    def preprocess_text(text):
        # Remove HTML tags
        try:
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text()
        except Exception as e:
            print(f"Error removing HTML tags: {e}")

        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Convert to lowercase
        text = text.lower()

        # Lemmatization using spaCy
        doc = nlp(text)
        lemmatized_text = ' '.join([token.lemma_ for token in doc])

        # Remove extra whitespaces
        lemmatized_text = re.sub(r'\s+', ' ', lemmatized_text).strip()

        return lemmatized_text

    faq_data['Question'] = faq_data['Question'].apply(preprocess_text)
    faq_data['Answer'] = faq_data['Answer'].apply(preprocess_text)
    vectorizer_distilbert = TfidfVectorizer(
        stop_words='english',
        max_df=0.85, # We are going to ignore the terms that appear in more than 85% of the documents
        min_df=2, # We are going to ignore the terms that appear in less than 2 documents
        ngram_range=(1, 2),  # Using unigrams(n=1 gram) and bigrams (n=2 gram)
    )

    distilbert_matrix = vectorizer_distilbert.fit_transform(faq_data['Question'] + faq_data['Answer'])

    # Function to get the answer to a question using similarity concept integrated with DistilBERT
    def get_answer_distilbert_like(question):
        question_distilbert_like = vectorizer_distilbert.transform([preprocess_text(question)])
        cosine_similarities_distilbert_like = linear_kernel(question_distilbert_like, distilbert_matrix).flatten()
        # Fetching the index of the most similar FAQ
        faq_index_distilbert_like = cosine_similarities_distilbert_like.argmax()
        return faq_data['Answer'].iloc[faq_index_distilbert_like]

    # Function to get the answer to a question using DistilBERT model
    def get_answer_distilbert(question, context):
        inputs = tokenizer_distilbert(question, context, return_tensors='pt')
        outputs = distilbert_model(**inputs)
        # Extract start and end logits from the output
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        # Get the answer span
        start_index = torch.argmax(start_logits, dim=1).item()
        end_index = torch.argmax(end_logits, dim=1).item() + 1
        # Get the answer text from the context
        answer = tokenizer_distilbert.decode(inputs['input_ids'][0][start_index:end_index])

        return answer

    # Function to handle user queries using our trained model
    def answer_user_query():
        print("Welcome to the University FAQ Question Answering System! Feel free to ask any questions about the university.")

        while True:
            user_question = input("🎓  Please Enter your question (or type 'q' to quit): ").strip().lower()

            if user_question == 'q':
                print("Thank You for using the Question Answering System. Have a Nice Day!👋")
                break

            # Get the answer using integrated models (DistilBERT-like and DistilBERT)
            distilbert_like_answer = get_answer_distilbert_like(user_question)
            distilbert_answer = get_answer_distilbert(user_question, "dummy context")

            print("\n🌟  Answer:", distilbert_like_answer)
            print("\n" + "=" * 50 + "\n")
    answer_user_query()