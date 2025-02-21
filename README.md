# Review Sentiment Analysis

This project compares six methods for sentiment analysis on product reviews. The results show that **Support Vector Classification (SVC)** delivers the most accurate predictions.

## 🚀 Features
- ✅ Uses Amazon Fashion reviews (download [here](https://nijianmo.github.io/amazon/index.html)).
- ✅ Selects the first 6,000 reviews, keeping only 1-star and 5-star ratings.
- ✅ Applies text preprocessing with lemmatization, stopword removal, and Count Vectorization.
- ✅ Tests six classification models: Naive Bayes, Logistic Regression, Random Forest, SVC, XGBoost, and CNN.
- ✅ Balances the dataset by resampling 1-star reviews before training a CNN.

## 📊 Analysis
- ✅ Models like Naive Bayes, Logistic Regression, Random Forest, XGBoost, and CNN (without resampling) perform well for 5-star reviews, but they generate too many false positives.
- ✅ CNN with resampling and SVC improve performance by reducing false positives.
- ✅ SVC is also better at predicting true positives and achieves a higher accuracy than CNN with resampling.
- ✅  SVC is the best-performing model, reaching **93% accuracy**.

## 🔹 Codes
- ✅ [kfold_template.py](https://github.com/huiyuy0913/review_sentiment_analysis/blob/main/kfold_template.py) defines a function run_kfold() that performs K-Fold Cross-Validation on
a given machine learning model (machine). It evaluates the model using various metrics such as R² score, accuracy, confusion matrix, F1-score, precision, and recall.
- ✅ [five_methods.py](https://github.com/huiyuy0913/review_sentiment_analysis/blob/main/five_methods.py) sentiment classification on Amazon fashion reviews using various
 machine learning models, including naive baysian, logistic regression, random forest, SVC and XGBoost. It applies data preprocessing, vectorization, and K-Fold Cross-Validation to evaluate model performance.
- ✅ [cnn_without_resample.py](https://github.com/huiyuy0913/review_sentiment_analysis/blob/main/cnn_without_resample.py) performs binary sentiment classification on Amazon fashion reviews using a CNN.
- ✅ [cnn_with_resample.py](https://github.com/huiyuy0913/review_sentiment_analysis/blob/main/cnn_with_resample.py) performs binary sentiment classification on Amazon fashion reviews using a Convolutional Neural Network (CNN). It applies resampling to improve model performance, ensuring balanced training data.
