import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
import math
import sys
import os
import gc
import shutil
import glob 
from shutil import copyfile
import errno
import itertools
import time

####################################################################

import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import Queue

###########################################################################

def error_msg(num):
    if (num==1):
        print("\nERROR!: Command Line Arguments: \n1) input auto encoded file\n")
    elif(num==2):
        print("\nERROR!: Can not open input file\n")
    sys.exit()
    
####################################################################

def som_initialize(dimension, number_of_centroids):
    rows = int(math.sqrt(number_of_centroids)) 
    network_dimensions = np.array([rows, rows])
    cnet = np.random.random((network_dimensions[0], network_dimensions[1], dimension))
    cnet_count = np.zeros((network_dimensions[0], network_dimensions[1]))
    return cnet,cnet_count

####################################################################

if len(sys.argv)<2:
    error_msg(1)
input_file = sys.argv[1]
if(os.path.exists(input_file)==False):
    error_msg(2)

####################################################################

net,net_count = som_initialize(2, 1500)

####################################################################

def provide_changes(start_index, end_index, col_start_index, col_end_index, update_queue_in, update_queue_out):
    col = net.shape[1]
    while True:
        if(update_queue_in.qsize()==0):
            continue
        s = update_queue_in.get()
        if s is None:
            break
        else:
            r = s[0]
            l = s[1]
            bmu_idx = np.array([s[2][0],s[2][1]])
            t = np.array([s[3][0],s[3][1]])
            rad = r**2 
            for x in range(start_index,end_index):
                for y in range(col_start_index,col_end_index):
                    w = net[x][y]
                    w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
                    if w_dist < rad:                                     
                        item = (x, y, (w + (l * np.exp(-w_dist / (2*rad)) * (t - w))))                   
                        update_queue_out.put(item)                    
            update_queue_out.put(None)
            
####################################################################

def find_bmu(start_index, end_index, col_start_index, col_end_index, bmu_queue_in, bmu_queue_out):
    col = net.shape[1]
    while True:
        if(bmu_queue_in.qsize()==0):
            continue
        s = bmu_queue_in.get()
        if s is None:
            break
        else:
            t = np.array([s[0],s[1]])
            bmu_idx = (0,0)
            min_dist = np.iinfo(np.int).max
            for x in range(start_index,end_index):
                for y in range(col_start_index,col_end_index):
                    w = net[x][y]
                    sq_dist = np.sum((w - t) ** 2)
                    if sq_dist < min_dist:
                        min_dist = sq_dist
                        bmu_idx = (x,y)
            bmu_queue_out.put((bmu_idx, min_dist))
            bmu_queue_out.put(None)
            
####################################################################

def decay_radius(initial_radius, i, time_constant):
    return initial_radius * np.exp(-i / time_constant)

####################################################################

def decay_learning_rate(initial_learning_rate, i, n_iterations):
    return initial_learning_rate * np.exp(-i / n_iterations)

####################################################################


if __name__ == "__main__": 
    print("Size: " + str(net.shape[0]) + ", " + str(net.shape[1]) + "\n")
    n_iterations = 300
    batch_size = 10000
    init_learning_rate = 0.01
    init_radius = max(net.shape[0], net.shape[1]) / 2
    time_constant = n_iterations / np.log(init_radius)
    ####################################################################
    #num_workers = multiprocessing.cpu_count()
    total_blocks = 150
    num_workers = int(math.sqrt(total_blocks))
    if num_workers>net.shape[0]:
        num_workers = net.shape[0] 
    #num_workers = net.shape[0]
    per_part_size = (int)(net.shape[0]/num_workers)
    positions = []
    start = 0
    while True:
        end = start + per_part_size
        if end>=net.shape[0]:
            end = net.shape[0]
            positions.append((start,end))
            break
        positions.append((start,end))
        start = end
        if start>=net.shape[0]:
            break
    
    full_positions = []
    for row in range(len(positions)):
        for col in range(len(positions)):
            full_positions.append((positions[row][0],positions[row][1],positions[col][0],positions[col][1]))
        
    
    ####################################################################
    c = 0
    gc.collect()
    bmu_queue_in = multiprocessing.Queue()
    bmu_queue_out = multiprocessing.Queue()
    update_queue_in = multiprocessing.Queue()
    update_queue_out = multiprocessing.Queue()
    ####################################################################
    bmu_workers=[]
    ####################################################################
    p_count = 0
    for ind in full_positions:
        #print("\nBMU Process: " + str(ind[0]) + "," + str(ind[1]))
        tmp = multiprocessing.Process(target=find_bmu, args=(ind[0], ind[1], ind[2], ind[3], bmu_queue_in, bmu_queue_out, ))
        #tmp = multiprocessing.Process(target=find_bmu, args=(ind[0], ind[1], 0, (int)(net.shape[1]/2), bmu_queue_in, bmu_queue_out, ))
        #tmp2 = multiprocessing.Process(target=find_bmu, args=(ind[0], ind[1], (int)(net.shape[1]/2), net.shape[1], bmu_queue_in, bmu_queue_out, ))
        tmp.daemon=True
        #tmp2.daemon=True
        tmp.start()
        #tmp2.start()
        bmu_workers.append(tmp)
        p_count = p_count + 1
        #bmu_workers.append(tmp2)           
    ####################################################################
    update_workers=[]                            
    ####################################################################
    for ind in full_positions:
        #print("\nUpdate Process: " + str(ind[0]) + "," + str(ind[1]))
        tmp = multiprocessing.Process(target=provide_changes, args=(ind[0],ind[1], ind[2], ind[3], update_queue_in, update_queue_out,))
        #tmp = multiprocessing.Process(target=provide_changes, args=(ind[0],ind[1], 0, (int)(net.shape[1]/2), update_queue_in, update_queue_out,))
        #tmp2 = multiprocessing.Process(target=provide_changes, args=(ind[0],ind[1], (int)(net.shape[1]/2), net.shape[1], update_queue_in, update_queue_out,))
        tmp.daemon=True
        #tmp2.daemon=True
        tmp.start()
        #tmp2.start()
        update_workers.append(tmp)
        p_count = p_count + 1
        #update_workers.append(tmp2)
    print("Total # of Process: " + str(p_count))
    ####################################################################
    start = time.time()
    for i in range(n_iterations):
        gc.collect()
        print("Iteration: " + str(i))
        data = []
        count = 0
        with open(input_file) as f:
            for line in f:
                if line!='' and line!=' ' and line!='\n':
                    line = line.strip().split()
                    row_number = int(line[0])
                    x_cord = float(line[1])
                    y_cord = float(line[2])
                    data.append((row_number,x_cord,y_cord))
                    count = count + 1
                    if count>batch_size:                    
                        for item in data:
                            row_num = item[0]
                            data_item = (item[1],item[2])
                            c = c + 1
                            for post in range(len(bmu_workers)):
                                bmu_queue_in.put(data_item)
                            ####################################################################
                            min_dist = np.iinfo(np.int).max
                            bmu =  None
                            bmu_idx = None
                            nun_count = 0
                            while True:
                                if nun_count==len(bmu_workers):
                                    break
                                if bmu_queue_out.qsize()>0: 
                                    s = bmu_queue_out.get()
                                    if s is None:
                                        nun_count = nun_count + 1
                                    else:
                                        if s[1]<min_dist:
                                            min_dist = s[1]
                                            bmu_idx = s[0]
                            old_count = net_count[bmu_idx[0]][bmu_idx[1]]
                            net_count[bmu_idx[0]][bmu_idx[1]] = old_count + 1 
                            #print("\n" + str(c) + ": " + str(item[1]) + " " + str(item[2]) + " bmu_idx: " + str(bmu_idx))
                            ####################################################################                        
                            r = decay_radius(init_radius, i, time_constant)
                            l = decay_learning_rate(init_learning_rate, i, n_iterations)
                            for post in range(len(update_workers)):
                                update_queue_in.put((r,l,bmu_idx,data_item))
                            update_list = []
                            nun_count = 0
                            while True:
                                if nun_count==len(update_workers):
                                    break 
                                s = update_queue_out.get()
                                if s is None:
                                    nun_count = nun_count + 1
                                else:
                                    update_list.append(s)
                            for u in update_list:
                                net[u[0]][u[1]] = u[2]                             
                            ####################################################################
                        count = 0
                        data = []
        if count>0:                    
            for item in data:
                row_num = item[0]
                data_item = (item[1],item[2])
                c = c + 1
                for post in range(len(bmu_workers)):
                    bmu_queue_in.put(data_item)
                ####################################################################
                min_dist = np.iinfo(np.int).max
                bmu =  None
                bmu_idx = None
                nun_count = 0
                while True:
                    if nun_count==len(bmu_workers):
                        break
                    if bmu_queue_out.qsize()>0: 
                        s = bmu_queue_out.get()
                        if s is None:
                            nun_count = nun_count + 1
                        else:
                            if s[1]<min_dist:
                                min_dist = s[1]
                                bmu_idx = s[0]
                old_count = net_count[bmu_idx[0]][bmu_idx[1]]
                net_count[bmu_idx[0]][bmu_idx[1]] = old_count + 1 
                ####################################################################                        
                r = decay_radius(init_radius, i, time_constant)
                l = decay_learning_rate(init_learning_rate, i, n_iterations)
                for post in range(len(update_workers)):
                    update_queue_in.put((r,l,bmu_idx,data_item))
                update_list = []
                nun_count = 0
                while True:
                    if nun_count==len(update_workers):
                        break 
                    s = update_queue_out.get()
                    if s is None:
                        nun_count = nun_count + 1
                    else:
                        update_list.append(s)
                for u in update_list:
                    net[u[0]][u[1]] = u[2]                             
                ####################################################################
            count = 0
            data = []
        ####################################################################
        with open("som_grid.txt",'w') as f:
            for x in range(net.shape[0]):
                for y in range(net.shape[1]):
                    f.write(str(x) + " " + str(y) + " " + str(net[x][y][0]) + " " + str(net[x][y][1]) + "\n")
        with open("som_grid_count.txt",'w') as f:
            for x in range(net.shape[0]):
                for y in range(net.shape[1]):
                    f.write(str(x) + " " + str(y) + " " + str(net_count[x][y]) + "\n")
        with open("som_iteration.txt",'w') as f:
            f.write(str(n_iterations) + "\n" + str(i))
    #################################################################### 
    end = time.time()
    print("Duration: " + str(end-start))
    bmu_queue_in.put(None)
    update_queue_in.put(None)
    for worker in bmu_workers:
        if worker.is_alive()==True:
            worker.terminate()
            del worker
    for worker in update_workers:
        if worker.is_alive()==True:
            worker.terminate()
            del worker
    bmu_queue_in.close()
    bmu_queue_out.close()
    update_queue_in.close()
    update_queue_out.close()
    del bmu_queue_in
    del bmu_queue_out
    del update_queue_in
    del update_queue_out
    del bmu_workers
    del update_workers
    gc.collect()