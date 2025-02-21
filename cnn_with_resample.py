import pandas
import string

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from pandarallel import pandarallel
import tensorflow as tf 

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D 
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from sklearn.utils import resample

################################read reviews and stars#########################
dataset = pandas.read_csv('AMAZON_FASHION.csv', usecols=["overall", "verified", "reviewText"])

###############################clean data: rename columns and drop NaN rows##########################
dataset.rename(columns={'overall': 'stars', 'reviewText': 'text'}, inplace=True)
print(dataset.isna().sum())
dataset.dropna(subset=['text'], inplace=True)
print(dataset.isna().sum())
print(dataset)

################################choose reviews with 1 and 5 stars#########################
dataset = dataset[0:6000]
dataset = dataset[(dataset['stars']==1)|(dataset['stars']==5)]
dataset.reset_index(drop=True, inplace=True)
print(dataset.shape)

################################pre processing the data################################
lemmatizer = WordNetLemmatizer()
def pre_processing(text):
	text_processed = text.translate(str.maketrans('', '', string.punctuation))
	text_processed = text_processed.split()
	result = []
	for word in text_processed:
		word_processed = word.lower()
		if word_processed not in stopwords.words("english"):
			word_processed = lemmatizer.lemmatize(word_processed)
			result.append(word_processed)
	return result

pandarallel.initialize(progress_bar=True)
dataset['text'] = dataset['text'].parallel_apply(pre_processing)
dataset['text'] = dataset['text'].parallel_apply(lambda x: ' '.join(x))

print(dataset)




print("="*10 + "CNN with resample" + "="*10)
X = dataset['text']
y = dataset['stars']

split_number = 5
kfold_object = KFold(n_splits=split_number, shuffle=True, random_state=42)
kfold_object.get_n_splits(X)

results_accuracy = []
results_confusion = []
results_precision = []
results_recall = []
results_f1score = []

MAX_WORDS = 10000
MAX_LENGTH = 200

for training_index, test_index in kfold_object.split(X):
    X_training = X[training_index]
    y_training = y[training_index]
    X_test = X[test_index]
    y_test = y[test_index]

    train_data = pandas.DataFrame({'text': X_training, 'stars': y_training})
    data_majority = train_data[train_data['stars'] == 5]  
    data_minority = train_data[train_data['stars'] == 1]
    # print("majority class before upsample:",data_majority.shape)
    # print("minority class before upsample:",data_minority.shape)
    data_minority_upsampled = resample(data_minority, 
                                    replace=True,     # sample with replacement
                                    n_samples= data_majority.shape[0],    # to match majority class
                                    random_state=123) # reproducible results
    df_balance = pandas.concat([data_majority, data_minority_upsampled])
    df_balance.reset_index(drop=True, inplace=True)
    # print("After upsampling\n",df_balance.stars.value_counts(),sep = "")
    X_training = df_balance['text']
    y_training = df_balance['stars']


    token = Tokenizer()
    token.fit_on_texts(X_training)
    vocab = len(token.index_word) + 1
    # print("Vocabulary size={}".format(len(token.word_index)))
    # print("Number of Documents={}".format(token.document_count))
    X_training = token.texts_to_sequences(X_training)
    X_test = token.texts_to_sequences(X_test)
    X_training = pad_sequences(X_training, maxlen=MAX_LENGTH, padding='post') 
    X_test = pad_sequences(X_test, maxlen=MAX_LENGTH, padding='post')

    num_classes=2
    label_encoder = LabelEncoder()
    y_training = label_encoder.fit_transform(y_training)
    y_test = label_encoder.fit_transform(y_test)

    vec_size = 300
    model = Sequential()
    model.add(Embedding(vocab, vec_size, input_length=MAX_LENGTH))
    model.add(Conv1D(64,8, activation="relu"))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.1))
    model.add(Dense(8, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(4, activation='relu'))
    model.add(Dropout(0.1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
    # model.summary()
    model.fit(X_training, y_training,  batch_size=4, epochs=10, shuffle=True, validation_split=0.1, verbose=1)
    def predictions(x):
        prediction_probs = model.predict(x)
        predictions = [1 if prob > 0.5 else 0 for prob in prediction_probs]
        return predictions
    results_accuracy.append(accuracy_score(y_test, predictions(X_test)))
    results_confusion.append(confusion_matrix(y_test, predictions(X_test)))
    results_precision.append(precision_score(y_test, predictions(X_test)))
    results_recall.append(recall_score(y_test, predictions(X_test)))
    results_f1score.append(f1_score(y_test, predictions(X_test)))

    print(results_accuracy)
    print(results_precision)
    print(results_recall)
    print(results_f1score)
    for i in results_confusion:
        print(i)
    # print(predictions(X_test))