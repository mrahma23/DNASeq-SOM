from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib import path
import math
from sklearn.utils.extmath import softmax
import numpy as np
import matplotlib.patches as patches
import random
from sklearn import preprocessing
from numpy import ones,vstack
from numpy.linalg import lstsq
from scipy.integrate import simps
import sys


def error_msg(num):
    if (num==1):
        print("\nERROR!: Command Line Arguments: \n1) input auto encoded file\n")
    elif(num==2):
        print("\nERROR!: Can not open file\n")
    sys.exit()

if len(sys.argv)<2:
    error_msg(1)
input_file = sys.argv[1]

bounding_box = 20
########################LOAD NON-ZERO INDICES#####################
non_zero_indices = []
zero_indices = []
total = 0
with open("som_grid_count.txt", "r") as f:
    for line in f:
        line = line.strip().split()
        row = (int)(line[0])
        col = (int)(line[1])
        count = (float)(line[2])
        if count>0:
            non_zero_indices.append((row,col))
            total = total + 1
        else:
            zero_indices.append((row,col))

percentage = 15
k = len(zero_indices) * percentage // 100
indicies = random.sample(range(len(zero_indices)), k)
zero_counts = [zero_indices[i] for i in indicies]

####################LOAD SOM POINTS###############################
som_points = []
som_indices = []
with open("som_grid.txt","r") as f:
    for line in f:
        line = line.strip().split()
        row = (int)(line[0])
        col = (int)(line[1])
        x_cord = (float)(line[2])
        y_cord = (float)(line[3])
        item = (row,col)
        if (item in non_zero_indices) or (item in zero_counts):
            som_points.append([x_cord,y_cord])
            som_indices.append([row,col])
som_min_max_scaler = preprocessing.MinMaxScaler()
som_points = som_min_max_scaler.fit_transform(som_points)

##########################CREATE VORONOI DIAGRAM#################           
vor = Voronoi(som_points)
voronoi_plot_2d(vor, show_vertices=False)
plt.savefig("voronoi.png")
#print(vor.ridge_vertices)
som_points_edges = []
for i in range(len(vor.points)):
    current_edges = []
    voronoi_vertices_indices = vor.regions[vor.point_region[i]]
    if -1 in voronoi_vertices_indices:
        som_points_edges.append(None)
        continue
    else:
        current = []
        for item in voronoi_vertices_indices:
            p = vor.vertices[item].tolist()
            current.append((p[0],p[1]))
        p = path.Path(current)
        som_points_edges.append(p)
      
###########################LOAD DATA###########################
data = []
original_sequence_number = []
data_som_map = []
data_som_pdf = []

with open(input_file, "r") as f:
    for line in f:
        line = line.strip().split()
        orginial_seq = (int)(line[0])
        x_cord = (float)(line[1])
        y_cord = (float)(line[2])
        item = [x_cord,y_cord]
        data.append(item)
        probable_places = []
        original_sequence_number.append(orginial_seq)
        for i in range(len(vor.points)):
            dist = math.sqrt(sum([(a - b)**2 for a, b in zip(vor.points[i].tolist(), item)]))
            similarity = 1.00 / (1.00 + dist)
            probable_places.append((i,similarity))
        probable_places = sorted(probable_places, key = lambda x:x[1], reverse = True)
        stop = len(probable_places)
        if len(probable_places)>bounding_box:
            stop = bounding_box
        current_som_map = []
        current_som_pdf = []
        for it in probable_places[:stop]:
            current_som_map.append(it[0])
            current_som_pdf.append(it[1])
        current_som_pdf = softmax(np.array([current_som_pdf]))
        data_som_map.append(current_som_map)
        data_som_pdf.append(current_som_pdf[0].tolist())

min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(data)

####################Prune 1############################################
edge_m_c = []
for ridge in vor.ridge_vertices:
    points = [vor.vertices[ridge[0]].tolist(), vor.vertices[ridge[1]].tolist()]
    if -1 in ridge:
        edge_m_c.append(None)
    else:
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        edge_m_c.append((m,c))

cluster_reps = {}
for item in vor.points.tolist():
    cluster_reps[str(item)] = []
    
for d in range(len(data)):
    condidates = vor.points.tolist()
    remove_set = []
    point_indices = data_som_map[d]
    pdf = data_som_pdf[d]
    points = []
    for item in point_indices:
        p = vor.points[item].tolist()
        points.append((p[0],p[1]))
    for i in range(len(vor.points)):
        if som_points_edges[i] is None:
            continue
        results = som_points_edges[i].contains_points(points)
        allTrue = True
        for r in results:
            if r==False:
                allTrue = False
                break
        if allTrue==True:
            remove_set.append(vor.points[i].tolist())
        
    for rdg in range(len(vor.ridge_vertices)):
        if edge_m_c[rdg] is None:
            continue
        else:
            m = edge_m_c[rdg][0]
            c = edge_m_c[rdg][1]
            for ridge_ps in vor.ridge_points[rdg]:
                x = vor.points[ridge_ps][0]
                y = vor.points[ridge_ps][1]
                val = m*x -y + c
                oppositeSide = True
                for p in points:
                    cur_val = m*p[0] - p[1] + c
                    if val>=0.00 and cur_val>=0.00:
                        oppositeSide = False
                        break
                    elif val<0.00 and cur_val<0.00:
                        oppositeSide = False
                        break
                if oppositeSide==True:
                    remove_set.append([x,y])
    for rm in remove_set:
        if rm in condidates:
            condidates.remove(rm)
        
    min_area = 999999999
    min_candidate = None
    for cand in condidates:
        cand_distances = []
        for p in range(len(points)):
            pt = [points[p][0],points[p][1]]
            probability = pdf[p]
            cand_distances.append(math.sqrt(sum([(a - b)**2 for a, b in zip(cand, pt)]))*probability)
        area = simps(cand_distances, dx=2)
        if area<min_area: 
            min_area = area
            min_candidate = cand
    members = cluster_reps[str(min_candidate)]
    members.append(original_sequence_number[d])
    cluster_reps[str(min_candidate)] = members
    
with open("cluster_output.txt", "w") as f:
    for key,value in cluster_reps.items():
        if len(value)>0:
            for item in value:
                f.write(str(item) + ' ')
            f.write('\n')
    
                            
                        
                                 
                     
                
            


    
                
            


        




 
        