import sys
import gc
import os
import numpy as np
np.set_printoptions(threshold=np.nan)
import theano
import keras
from numpy.random import seed
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import LSTM, RepeatVector
from sklearn.preprocessing import minmax_scale
import time
import matplotlib.pyplot as plt
import itertools

###########################################################################

def error_msg(num):
    if (num==1):
        print("\nERROR!: Command Line Arguments: \n1) input_kmer_feature_files\n")
    elif(num==2):
        print("\nERROR!: Can not open input file\n")
    elif (num==3):
        print("\nERROR! Could not retrieve k from the file: value_of_k.txt\n")
    elif (num==4):
        print("\nlenght of mapper and predicted data did not match!\n")   
    sys.exit()

###########################################################################

if len(sys.argv)<2:
    error_msg(1)
input_file = sys.argv[1]
if(os.path.exists(input_file)==False):
    error_msg(2)
k = -1
try:
    with open("value_of_k.txt",'r') as f:
        for line in f:
            k = int(line)
            break
except:
    error_msg(3)
total_features = ncol = 4**k

###########################################################################

def generate_data(path):
    with open(path) as f:
        for line in f:
            if line!='' and line!=' ' and line!='\n':
                line = line.strip().split()
                row_number = int(line[0])
                data_items = [0.0]*total_features
                for item in line[1:]:
                    item = item.strip().split('-')
                    col_number = int(item[0])
                    data_value = float(item[1])
                    data_items[col_number] = data_value
                yield (row_number,data_items)
            
###########################################################################

def create_autoEncoder():
    input_dim = Input(shape=(ncol,))
    encoded = Dense(2, activation='relu')(input_dim)
    decoded = Dense(ncol, activation='sigmoid')(encoded)
    #########################################################
    autoencoder = Model(input_dim, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    #########################################################
    encoder = Model(input_dim, encoded)
    #########################################################
    return (autoencoder, encoder)

###########################################################################

def create_LSTM_autoEncoder():
    timesteps = 200
    latent_dim = 2
    input_dim = Input(shape=(ncol,))
    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(latent_dim)(inputs)
    #########################################################
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(input_dim, return_sequences=True)(decoded)
    #########################################################
    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    return (sequence_autoencoder, encoder)

###########################################################################

def main():
    autoencoder,encoder = create_autoEncoder()
    data = []
    full_data = []
    count = 0
    batch_size = 100
    per_batch_iteration = 2500 
    
    ##########################################################################################
    
    start = time.time()
    provider,provider_backup = itertools.tee(generate_data(input_file)) 
    for line in provider:
        data.append(line[1])
        full_data.append(line[1])
        count = count + 1
        if count>batch_size:
            for i in range(per_batch_iteration):
                autoencoder.train_on_batch(data,data,class_weight=None, sample_weight=None)
            count = 0
            data = []    
    if count>10:
        for i in range(per_batch_iteration):
            autoencoder.train_on_batch(data,data,class_weight=None, sample_weight=None)
        count = 0
        data = []
    end = time.time()
    
    ##########################################################################################
    
    data = []
    mapper = []
    count = 0
    batch_size = 5000
    output_file = open("two_dimensional_data.txt","w")
    for line in provider_backup:
        data.append(line[1])
        mapper.append(line[0])
        count = count + 1
        if count>batch_size:
            x = []
            y = []
            encoded_out = encoder.predict_on_batch(data)
            if len(encoded_out)==len(mapper):
                for i in range(len(mapper)):
                    x.append((encoded_out[i])[0])
                    y.append((encoded_out[i])[1])
                    output_file.write(str(mapper[i]) + ' ' + str((encoded_out[i])[0]) + ' ' + str((encoded_out[i])[1]) + '\n')
                plt.scatter(x, y, s=0.5)                
            else:
                error_msg(4)
            count = 0
            data = []
            mapper = []
            encoded_out = []
    if count>0:
        x = []
        y = []
        encoded_out = encoder.predict_on_batch(data)
        if len(encoded_out)==len(mapper):
            for i in range(len(mapper)):
                x.append((encoded_out[i])[0])
                y.append((encoded_out[i])[1])
                output_file.write(str(mapper[i]) + ' ' + str((encoded_out[i])[0]) + ' ' + str((encoded_out[i])[1]) + '\n')
            plt.scatter(x, y, s=0.5)
        else:
            error_msg(4)
        count = 0
        data = []
        mapper = []
        encoded_out = []
    output_file.close()

    print("Error: " + str(autoencoder.evaluate(full_data, full_data, batch_size=1, verbose=0, sample_weight=None)))
    print("Time:" + str(end-start))
    plt.scatter(x, y, s=0.5)
    plt.savefig("plot.png")
    plt.show()
    

if __name__ == "__main__": 
    main()