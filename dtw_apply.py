from dtw import *
from datetime import datetime
import os
import random
import numpy as np

dir_des_path = "/home/jane/Downloads/xulytiengnoi/bt2/res/mfccs/"


random.seed(datetime.now())

lables = ('0', '1', '2', '3', '4', '5', 
	'6', '7', '8', '9', 'lam', 'linh', 
	'm1', 'mot', 'muoi', 'nghin', 'sil',
	'tram', 'trieu', 'tu')

ts = {}

for lable in lables:
	print("Dang lay mau label {}".format(lable))
	dir_path = os.path.join(dir_des_path, lable)
	txt_file_name = '{}.txt'
	t1 = np.loadtxt(os.path.join(dir_path, txt_file_name.format(0)))
	t2 = np.loadtxt(os.path.join(dir_path, txt_file_name.format(1)))
	t3 = np.loadtxt(os.path.join(dir_path, txt_file_name.format(3)))
	ts[lable] = (t1, t2, t3)

count = 0

for i in range(200):
	

	query_label = lables[random.randrange(len(lables))]
	query_path = os.path.join(dir_des_path, query_label, txt_file_name.format(random.randrange(10)))

	query = np.loadtxt(query_path)

	query = np.transpose(query)

	#print(query_label)

	min_dis = 100000
	lable_query = '?'

	for lable in lables:
		for t in ts[lable]:
			alignmentOBE = dtw(query, np.transpose(t), 
				keep_internals=True, 
				step_pattern=asymmetric,
				open_end=True, open_begin=True)
			if alignmentOBE.normalizedDistance < min_dis:
				min_dis = alignmentOBE.normalizedDistance
				lable_query = lable

	#print(lable_query)

	print("Thuc te: {} va ket qua du doan: {}".format(query_label, lable_query))

	if (lable_query == query_label):
		count += 1

print("Ty le dung")
print(count/200)

