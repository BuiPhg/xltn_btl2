import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd
import os
import scipy.io
import parse
import functools

dir_wavs = "/home/jane/Downloads/xulytiengnoi/bt2/res/wavs"
dir_mfccs = "/home/jane/Downloads/xulytiengnoi/bt2/res/mfccs"
dir_splited_wavs = "/home/jane/Downloads/xulytiengnoi/bt2/res/splited_wavs"
label_list = ('0', '1', '2', '3', '4', '5', 
	'6', '7', '8', '9', 'lam', 'linh', 
	'm1', 'mot', 'muoi', 'nghin', 'sil',
	'tram', 'trieu', 'tu')

dic = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'lam', 'linh', 'm1', 'mot', 'muoi', 'nghin', 'sil', 'tram', 'trieu', 'tu')

def getDirPaths():
	return [os.path.join(dir_mfccs, key) for key in label_list]

def makeDirs(dir_paths):
	[os.makedirs(path) for path in dir_paths if not os.path.exists(path)]

def getCouplePaths(label):
	return (os.path.join(dir_splited_wavs, label), os.path.join(dir_mfccs, label))

def printFilePerLabel(label):
	dir_splited_wav, dir_mfcc = getCouplePaths(label)
	# print(dir_splited_wav)
	# print(dir_mfcc)
	i = 0

	print('dang in mfcc cua {}'.format(label))

	for file in os.listdir(dir_splited_wav):
		wav_path = os.path.join(dir_splited_wav, file)
		#print(wav_path)
		signal, sr = librosa.load(wav_path)
		if (len(signal) < 2048):
			continue
		mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
		#print(mfccs.shape)
		delta_mfccs = librosa.feature.delta(mfccs, width=5)
		delta2_mfccs = librosa.feature.delta(mfccs, width=5, order = 2)
		mfccs_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
		mfccs_file = '{}.txt'.format(i)
		i = i + 1
		mfccs_path = os.path.join(dir_mfcc, mfccs_file)
		#print(mfccs_path)
		np.savetxt(mfccs_path, mfccs_features, fmt="%d")

def main():
	makeDirs(getDirPaths())
	[printFilePerLabel(label) for label in label_list]

main()