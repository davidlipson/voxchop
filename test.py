import librosa
import librosa.display
import numpy as np
import math
import matplotlib.pyplot as plt
import os

keys = ["c", "c+", "d", "eb", "e", "f", "f+", "g", "ab", "a", "b"]

def stft2chroma(fn):
	y, sr = librosa.load(fn)
	print(sr)
	S = np.abs(librosa.stft(y))
	chroma = librosa.feature.chroma_stft(S=S, sr=sr)
	return y, sr, chroma

# returns note clusters of 1. power
def clean(y, sr, chroma):
	# first pass: just use power=1.0 samples
	chroma = np.array([[1. if i >= 1. else 0. for i in c] for c in chroma])
	return chroma

# return timestmps/smple indices for beginning/ending of notes
def find(y, sr, chroma):
	notes = []
	for c in chroma:
		samples = []
		playing = False
		start = 0
		for index, x in enumerate(c):
			if x == 1. and playing == False:
				playing = True
				start = index
			elif x != 1. and playing == True:
				playing = False
				samples.append((start*512, index*512))
		notes.append(samples)
	return np.array(notes)

def plot(chroma):
	plt.figure(figsize=(10, 4))
	librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
	plt.colorbar()
	plt.title('Chromagram')
	plt.tight_layout()
	plt.show()

def save(output, y, sr):
	librosa.output.write_wav(output, y, sr)

# convert
t, tsr, tc = stft2chroma("test.wav")
print(len(t), len(tc[0]), len(tc))

# clean the chroma sequences
tc = clean(t, tsr, tc)
print(tc)

# find notes
notes = find(t, tsr, tc)
print(notes)

plot(tc)

for nid, n in enumerate(notes):
	for index, x in enumerate(n):
		if(x[1]-x[0] >= 5120):
			d = "out/%s/" % keys[nid]
			if not os.path.exists(d):
			    os.mkdir(d)
			save("%s%d.wav" % (d, index), t[x[0]:x[1]], tsr)

# test
# save("out.wav", t[0:int(len(t)/10)], tsr)
