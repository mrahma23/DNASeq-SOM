import numpy as np
from BTrees.OOBTree import BTree

####################################################################

import os
import gc
import shutil
import glob 
from shutil import copyfile
import errno
import sys
import itertools
import time
import math

####################################################################

import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import Queue

####################################################################

def error_msg():
	print "\nERROR!: Command Line Arguments: \n1) value_of_k [Integer value within 2-5]\n2) input_file [fasta file]\n"
	sys.exit()

####################################################################

if len(sys.argv)<3:
	error_msg()
k = -1
try:
	k = int(sys.argv[1])
except:
	error_msg()
if k<2 or k>5:
	error_msg()
with open("value_of_k.txt",'w') as f:
	f.write(str(k))
input_file = sys.argv[2]
output_file = input_file.split('.')[0] + '.txt' 

####################################################################

allowed = ['a','c','g','t']
features = BTree()
feature_count = -1
for ft_string in itertools.imap(''.join, itertools.product('acgt', repeat=k)):	
	feature_count = feature_count + 1
	features[ft_string]=feature_count
	
####################################################################	

def reader(input_file, inq):
    count = 0
    sequence_number=0
    f = open(input_file, 'r')
    sequence=''
    for line in f:
		line=line.strip()
		if line!='' and line!='\n' and line!=' ':
			if line.startswith('>'):
				if sequence!='':
					item=(sequence_number,sequence)
					inq.put(item)
					count = count + 1 
					sequence_number = sequence_number + 1
					sequence=''
			else:
				sequence=sequence+line    
    if sequence!='':
        item=(sequence_number,sequence)
        inq.put(item)
        count = count + 1   
    f.close()
    with open("total_number_sequences.txt", 'w') as f:
    	f.write(str(count))

####################################################################

def modifier(inq, outq):
	while True:
	    item=inq.get()
	    if item is None:
	        break
	    else:
			sequence_number=item[0]
			sequence=item[1].strip().lower()
			sequence = ''.join([i for i in sequence if i in allowed])
			local_features = BTree()
			for i in range(len(sequence)-k+1):
				kmer = sequence[i:i+k]
				feature_index = features[kmer]
				index = feature_index
				try:					
					current_count = local_features[index] 
					local_features[index] = current_count + 1
				except KeyError:
					local_features[index] = 1            
			#normalization
			sum = 0
			for value in local_features.values():
				sum = sum + (value*value)
			sum = math.sqrt(sum)
			for key in local_features.keys():
				value = local_features[key]
				local_features[key] = float(value)/sum			
			out_string = str(sequence_number)
			for key,val in local_features.iteritems():
				out_string = out_string + ' ' + str(key) + '-' + str(val)
			out_string = out_string + '\n'
			outq.put(out_string)
			del local_features
 
####################################################################

def writer(output_file, outq):
    f = open(output_file, 'w')
    while True:
        s = outq.get()
        if s is None:
            break
        else:
            f.write(s)
    f.close()
    
####################################################################

def main():
	num_workers = multiprocessing.cpu_count()
	workers=[]		
	inq = multiprocessing.Queue()
	outq = multiprocessing.Queue()
	####################################################################
	for i in range(num_workers):
		tmp = multiprocessing.Process(target=modifier, args=(inq, outq, ))
		tmp.daemon=True
		tmp.start()
		workers.append(tmp)	   
	####################################################################
	fileWriteProcess=multiprocessing.Process(target=writer, args=(output_file, outq,))
	fileWriteProcess.daemon=True
	fileWriteProcess.start()	
	####################################################################
	fileReadProcess=multiprocessing.Process(target=reader, args=(input_file, inq,))
	fileReadProcess.daemon=True
	fileReadProcess.start()	
	####################################################################
	fileReadProcess.join()
	if fileReadProcess.is_alive()==True:
		fileReadProcess.terminate()
		del fileReadProcess
	####################################################################
	for i in range(num_workers):
		inq.put(None)			
	####################################################################
	for worker in workers:
		worker.join()
	####################################################################
	for worker in workers:
		if worker.is_alive()==True:
			worker.terminate()
			del worker		  
	####################################################################
	outq.put(None)		
	####################################################################
	fileWriteProcess.join()		
	if fileWriteProcess.is_alive()==True:
		fileWriteProcess.terminate()
		del fileWriteProcess	  
	####################################################################			
	inq.close()
	outq.close()
	inq.join_thread()
	outq.join_thread()
	####################################################################	
	del num_workers
	del workers
	del inq
	del outq
	gc.collect()
	####################################################################

if __name__ == "__main__": 
    main()
