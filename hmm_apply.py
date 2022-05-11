import hmmlearn
import os
import random
import numpy as np
from hmmlearn import hmm
from sklearn.model_selection import train_test_split

dir_res_des_path = "/home/jane/Downloads/xulytiengnoi/bt2/res/mfccs/"

labels = ('0', '1', '2', '3', '4', '5', 
	'6', '7', '8', '9', 'lam', 'linh', 
	'm1', 'mot', 'muoi', 'nghin', 'sil',
	'tram', 'trieu', 'tu')

# hàm lấy dữ liệu
def getData():
	dic = {}
	for label in labels:
		dic[label] = []
		for file in os.listdir(os.path.join(dir_res_des_path, label)):
			a_data = np.loadtxt(os.path.join(dir_res_des_path, label, file))
			a_data = np.transpose(a_data)
			dic[label].append(a_data)

	return dic

def getXY(data):
	#pass

	X = {}
	Y = {}

	for label in labels:
		a_data = data[label]
		train_size = int(0.8 * len(a_data))
		random.shuffle(a_data)
		a_X = a_data[:train_size]
		a_Y = a_data[train_size:]

		a_length = [len(x) for x in a_X]
		a_X_real = np.concatenate(a_X)
		X[label] = (a_X_real, a_length)
		Y[label] = a_Y

	return (X, Y) 



# hàm chia dữ liệu thành train và test set
	


def train(X_total):
	GMMHMM_Models = {}

	for label in labels: 
		print('Dang huan luyen mo hinh cho label {}'.format(label))
		remodel = hmm.GMMHMM(n_components=3, n_mix = 3, n_iter=20)
		#remodel.covars_prior = np.array([0.02, 0.01, 0.01, 0.02])
		#remodel.covars_weight = np.array([0.25, 0.5, 0.25, 0.25])
		X, length = X_total[label]
		remodel.fit(X, lengths = length)
		GMMHMM_Models[label] = remodel

	return GMMHMM_Models

def predictRate(model, Y_total):

	count_total = 0;
	count_right = 0;

	for label in labels:
		Y = Y_total[label]
		scoreList = {}
		count_total += 1;
		for model_label in model.keys():
			a_model = model[model_label]
			score = a_model.score(Y[0])
			scoreList[model_label] = score
		predict = max(scoreList, key=scoreList.get)
		
		if predict == label:
			count_right += 1;

	return count_right / count_total;

def main():

	data = getData()

	# for label in labels:
	# 	print(len(data[label]))
	# 	print(data[label])
	# 	print()

	#getXY(data)

	X_total, Y_total = getXY(data)
	model = train(X_total)
	print('Ti le dung la:')
	print(predictRate(model, Y_total))

main()