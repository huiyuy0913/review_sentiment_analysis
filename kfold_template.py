from sklearn.model_selection import KFold
from sklearn import metrics


def run_kfold(data, target, machine, n, use_r2=True, use_accuracy=False, use_confusion=False, use_f1score = False, use_precision = False, use_recall = False):

	kfold_object = KFold(n_splits=n)
	kfold_object.get_n_splits(data)
	
	all_return_values = []
	i = 0
	for train_index, test_index in kfold_object.split(data):
		i=i+1
		
		data_train = data[train_index]
		target_train = target[train_index]
		data_test = data[test_index]
		target_test = target[test_index]

		machine.fit(data_train, target_train)

		prediction = machine.predict(data_test)
		return_value = []
		if (use_r2 == True):	
			r2 = metrics.r2_score(target_test, prediction)
			# print("R square score: ",r2)
			return_value.append(r2)
		if (use_accuracy == True):
			accuracy = metrics.accuracy_score(target_test, prediction)
			# print("Accuracy score: ",accuracy)
			return_value.append(accuracy)
		if (use_confusion == True):
			confusion = metrics.confusion_matrix(target_test, prediction)
			# print("Confusion matrix:\n ", confusion)
			return_value.append(confusion)
		if (use_f1score == True):
			f1score = metrics.f1_score(target_test, prediction)
			return_value.append(f1score)
		if (use_precision == True):
			precision = metrics.precision_score(target_test, prediction)
			return_value.append(precision)
		if (use_recall == True):
			recall = metrics.recall_score(target_test, prediction)
			return_value.append(recall)
		# print("\n\n") 
		all_return_values.append(return_value)
	return all_return_values


if __name__ == '__main__':
    run_kfold()
