import pandas
import string
import kfold_template

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb


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
data = dataset['text']
target = dataset['stars']
num_classes=2
label_encoder = LabelEncoder()
target = label_encoder.fit_transform(target)

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


count_vectorize_transformer = CountVectorizer(analyzer=pre_processing).fit(data)
data = count_vectorize_transformer.transform(data)


print("="*10 + "MultinomialNB" + "="*10)
machine = MultinomialNB() # naive baysian
machine.fit(data,target)
results = kfold_template.run_kfold(data, target, machine, 5, False, True, True, True, True, True)
for i in range(5):
	print([row[i] for row in results])


print("="*10 + "Logistic Regression" + "="*10)
machine = LogisticRegression(max_iter=100)
machine.fit(data,target)
results = kfold_template.run_kfold(data, target, machine, 5, False, True, True, True, True, True)
for i in range(5):
	print([row[i] for row in results])



print("="*10 + "Random Forest" + "="*10)
machine = RandomForestClassifier(
	                            random_state=42, 
                                max_features='sqrt',
                                bootstrap=True
								)
machine.fit(data,target)
results = kfold_template.run_kfold(data, target, machine, 5, False, True, True, True, True, True)
for i in range(5):
	print([row[i] for row in results])

################################ select the best max_feature before applying random forest (sqrt is the best)################################
# trials = []
# for w in [None, 'log2', 'sqrt']:
#     machine = RandomForestClassifier(random_state=42,max_features=w, bootstrap=True)
#     return_values = kfold_template.run_kfold(data, target, machine, 5, True, True, True, True, True, True)
#     all_r2 = [i[0] for i in return_values]
#     average_r2 = sum(all_r2)/len(all_r2)
#     trials.append((average_r2,w))
# trials.sort(key = lambda x: x[0], reverse = True)
# print(trials[:5])

print("="*10 + "SVC Linear" + "="*10)
machine = SVC(kernel='linear', probability=True)
machine.fit(data,target)
results = kfold_template.run_kfold(data, target, machine, 5, False, True, True, True, True, True)
for i in range(5):
	print([row[i] for row in results])


print("="*10 + "XGBoost" + "="*10)
machine = xgb.XGBClassifier()
machine.fit(data,target)
results = kfold_template.run_kfold(data, target, machine, 5, False, True, True, True, True, True)

for i in range(5):
	print([row[i] for row in results])



