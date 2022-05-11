import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd
import os
import scipy.io
import parse
import functools

dir_wavs = "/home/jane/Downloads/xulytiengnoi/bt2/res/wavs"
dir_splited_wavs = "/home/jane/Downloads/xulytiengnoi/bt2/res/splited_wavs"
label_list = ('0', '1', '2', '3', '4', '5', 
	'6', '7', '8', '9', 'lam', 'linh', 
	'm1', 'mot', 'muoi', 'nghin', 'sil',
	'tram', 'trieu', 'tu')

# tra ve path cua audio va label file ung voi ten file
def paths():
	dirs = [os.path.join(dir_wavs, dire) for dire in os.listdir(dir_wavs)]
	paths = [os.path.join(dire, file) for dire in dirs for file in os.listdir(dire)]
	wav_paths = [path for path in paths if path.endswith(".wav")]
	paths = [(wav_path, wav_path.replace('.wav', '.txt')) for wav_path in wav_paths if wav_path.replace('.wav', '.txt') in paths]
	return paths

def readLabelFile(label_path):
	with open(label_path) as file:
		lines = file.readlines()
	return lines

def readAudioFile(audio_path):
	return scipy.io.wavfile.read(audio_path)

# label_file thanh label_list
def loadLabelList(lines, rate):
	lines = [x.split() for x in lines]
	lines = [x for x in lines if len(x) == 3 and x[2] in label_list]
	#print(lines[0])
	index = lambda time : round(float(time) * rate)
	indexs = lambda x : (index(x[0]), index(x[1]), x[2])
	return map(indexs, lines)

# tra vef dic cac wav file
def splitAudio(audio, label_list): 
	split = lambda start, end : audio[start : end]
	addA = lambda dic, a : dic.setdefault(a[2], []).append(split(a[0], a[1])) or dic
	return functools.reduce(addA, label_list, {})

def loadDic(couple_path):
	print("Dang load file {}".format(couple_path[0]))
	audio_path, label_path = couple_path
	lines = readLabelFile(label_path)
	rate, audio = readAudioFile(audio_path)
	label_list = loadLabelList(lines, rate)
	dic = splitAudio(audio, label_list)
	return dic

def updateDic(dic_des, dic_res):
	[dic_des.setdefault(key, []).extend(dic_res[key]) for key in dic_res]

def loadTotalDic():
	total_dic = {}
	[updateDic(total_dic, loadDic(path)) for path in paths()]
	return total_dic

# print toltal_dic
def printDic(dic, rate):
	dir_paths = [os.path.join(dir_splited_wavs, key) for key in dic.keys()]
	[os.makedirs(path) for path in dir_paths if not os.path.exists(path)]

	for key in dic.keys():
		print("Dang in dir {}".format(key))
		list_data = dic[key]
		len_list_data = len(list_data)
		path_key = os.path.join(dir_splited_wavs, key, '{}.wav')
		for i in range(len_list_data):
			data = list_data[i]
			scipy.io.wavfile.write(path_key.format(i), rate, data)


def main(): 
	total_dic = loadTotalDic()
	printDic(total_dic, 22050)



main()		
