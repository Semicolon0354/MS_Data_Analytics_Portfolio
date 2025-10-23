# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:25:56 2024

@author: jdhum
"""


from transformers import BartTokenizer, BartForConditionalGeneration
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import PyPDF2




tokenizer=BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def extract_text_from_pdf(pdf_path):
    # Open the PDF file in read-binary mode
    with open(pdf_path, 'rb') as pdf_file:
        # Create a PdfReader object instead of PdfFileReader
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def classify_input(user_input):
    keywords = ['search', 'find', 'lookup', 'query']
    if any(keyword in user_input.lower() for keyword in keywords):
        return 'search'
    return 'not_search'

def perform_search(query, document_library):
    texts = [extract_text_from_pdf(doc) for doc in document_library]
    vectorizer = TfidfVectorizer()
    docs_tfidf = vectorizer.fit_transform(texts)
    query_tfidf = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_tfidf, docs_tfidf).flatten()
    top_doc = np.argmax(similarity_scores)
    return document_library[top_doc]

def summarize_text(doc):
    text=extract_text_from_pdf(doc)
    inputs = tokenizer([text],max_length=1024, truncation=True, return_tensors='pt')
    summary_ids =  model.generate(inputs['input_ids'], num_beams=5, max_length=200, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
    
def main(user_input, document_library):
    input_type = classify_input(user_input)
    if input_type == 'search':
        search_results = perform_search(user_input, document_library)
        summary = summarize_text(search_results)
        return summary
    else:
        return "This doesn't seem like a search query."

# Example usage
doc_names=['ISSW2024_O1.4.pdf','ISSW2024_O1.5.pdf','ISSW2024_O1.6.pdf']
document_library=[]
for i in range(len(doc_names)-1):
    document_library.append(os.path.join('Avalanche_Docs',doc_names[i]))

user_input = input("How can I help you?\n")
response = main(user_input, document_library)
print(response)
