# Bias-detection-in-text
This project I did for my NLP course in 6th semester. I tried to re-implement and add to the studies done in paper "Dbias: Detecting Biases and Ensuring Fairness in News Articles"  
I took MBIC dataset from Kaggle, did embeddings with BERT model and trained few RNNs models.  
A binary classifier that classified between "biased" and "unbiased" news article, and another multiclass classifier that also detected "opinion" of news reporter.  
Despite the limitation of dataset 1500 articles, the models were quite well classifying the news articles, for example the binary classifier had 84% G-AUC score, multiclass had >65% accuracy score on test set. The multiclass data had too much "opinion" data which was previously either biased or mostly unbiased, so the unbiased class got unbalanced in training set.  
I tested the same test set with simple prompt template and few shot learning with Google Gemini 1.5 flash model and found the trained model were performing better than the Google LLM.

