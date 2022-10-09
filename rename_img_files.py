import os
import random 
from glob import glob
#files = os.listdir()
files = glob('*.jpg')
files = sorted(files)
files = sorted(files, key = lambda x  : int(x.split('_')[0]))
timestamp ='123456784'
date = '08-10-2022'
time='09-22-31'
for i, file in enumerate(files):
	tag = i//2 + 1
	idx = i%2
	os.rename(file, str(timestamp) + '_' + str(tag) + '_' + date +'_' + time + '_' + str(idx) + '.jpg')


