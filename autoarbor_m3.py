#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoArbor_M3: The method 3 of AutoArbor Algorithm
Created on Feb 13 10:22:52 2020
Last revision: March 24, 2020

@author: Hanchuan Peng
"""

print(__doc__)

import time

import csv
import pandas
import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import os

from sklearn import cluster
#from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

#from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

import argparse

#some functions

def generate_tmp_swc(filename):
    #generate a temporary swc file which removes the # lines and replace the 
    #column index row
    fin = open(filename)
    fout = open("./tmp_swc.csv", "wt")
    fout.write( 'id type x y z r pid\n' )
    for line in fin:
        if not ( line.startswith('#') or (not line[0].isdigit()) ):
            ### ESWC format
            linesplit=line.split(' ')
            n=len(linesplit)
            if n>7:
                new_line=''
                for i in range(7):
                    new_line=new_line+' '+linesplit[i]
                new_line=new_line.lstrip()
                new_line=new_line+'\n'
                fout.write( new_line )
            else:
                fout.write( line )
    fin.close()
    fout.close()


def s_clustering(XS, coords, my_n_clusters):

    np.random.seed(0)
    
    # #############################################################################
    # Compute clustering 
    
    spectral = cluster.SpectralClustering(
        n_clusters=my_n_clusters, eigen_solver='arpack',random_state=0,
        affinity="precomputed")

    t0 = time.time()
    spectral.fit(XS)
    t_batch = time.time() - t0
    print(t_batch)

    my_labels = spectral.labels_
    print(my_labels)
    
    print("finish clustering")
    
    score = 0
    for k in zip(range(my_n_clusters)):
#        print(k)
        my_members = my_labels == k
        cluster_center = np.mean(coords[my_members, :], axis=0, dtype=np.float64) 
        cluster_std = np.std(coords[my_members, :], axis=0, dtype=np.float64)
#        print(cluster_std) 
        score += LA.norm(cluster_std)        
 
    print("finish s_function")
    return [score/my_n_clusters, my_labels];  

def avg_path(points,soma):
    # children-parent matrix
    original_id=points.index.values
    points=points.reset_index(drop=True)
    idlist=[]
    plist=[]
    tiplist=[]
    children=[[] for i in range(len(points))]
    #print("avg_path")
    #pandas.set_option('display.max_rows',100)
    #print(points)
    for i in range(len(points)):
        idlist.append(points['id'][i])     
        plist.append(points['pid'][i])

    for i in range(len(points)):
        pid=idlist.index(points['pid'][i]) if points['pid'][i] in idlist else -1
        if pid < 0:
            continue
        children[pid].append(i)

    tipN=0
    for i in range(len(points)):
        if len(children[i])==0:
            tipN=tipN+1
            tiplist.append(i)
               
    total_length=0
    for i in range(tipN):
        dx=points['x'][tiplist[i]]-soma['x'][0]
        dy=points['y'][tiplist[i]]-soma['y'][0]
        dz=points['z'][tiplist[i]]-soma['z'][0] 
        tmp_d=math.sqrt(dx*dx+dy*dy+dz*dz)
        total_length=total_length+tmp_d

    avg_length=total_length/tipN 
    return avg_length   

    
def child_parent(points):
    points=points.reset_index(drop=True)
    idlist=[]
    plist=[]
    children=[[] for i in range(len(points))]
    for i in range(len(points)):
        idlist.append(points['id'][i])     
        plist.append(points['pid'][i])
    for i in range(len(points)):
        pid=idlist.index(points['pid'][i]) if points['pid'][i] in idlist else -1
        if pid < 0:
            continue
        children[pid].append(i)
    return children


def compute_dis(df,a,b):
    dx=df['x'][a]-df['x'][b]
    dy=df['y'][a]-df['y'][b]
    dz=df['z'][a]-df['z'][b]
    dist=math.sqrt(dx*dx+dy*dy+dz*dz)
    return dist

def compute_shortest_path(points,soma):
    points=points.reset_index(drop=True)
    min_d=1000
    for i in range(len(points)):
        dx=points['x'][i]-soma['x'][0]
        dy=points['y'][i]-soma['y'][0]
        dz=points['z'][i]-soma['z'][0] 
        tmp_d=math.sqrt(dx*dx+dy*dy+dz*dz)
        if tmp_d<min_d:
            min_d=tmp_d
    return min_d


def split(points,soma):
    real_id=points.index.values
    points=points.reset_index(drop=True)
    children=child_parent(points)
    idlist=[]
    for i in range(len(points)):
        idlist.append(points['id'][i])     
    distance=[-1 for i in range(len(points))]
    soma_id=soma.index.values[0]
    stack=[]
    stack.append(soma_id)
    seen=set()
    seen.add(soma_id)
    distance[soma_id]=0
    while (len(stack)>0):
        vertex=stack.pop()
        nodes=children[vertex]
        for w in nodes:
            if w not in seen:
                distance[w]=distance[vertex]+compute_dis(points,vertex,w)
                stack.append(w)
                seen.add(w)
    d_index={}
    for i in range(len(distance)):
        d_index[distance[i]]=i
    distance.sort(reverse=True)
    print("longest")
    longest_id=d_index[distance[0]]
    print(longest_id)
    print(distance[0])
    
    # find the 7:3 point
    d_tmp=distance[0]
    stp=longest_id
    while (d_tmp>0.3*distance[0]):
        pid=idlist.index(points['pid'][stp]) if points['pid'][stp] in idlist else -1 
        if pid == -1:
            break   
        d_tmp=d_tmp-compute_dis(points,pid,stp)
        stp=pid
    print("7:3")

    # RETURN ids
    out=[]
    stack=[]
    stack.append(stp)
    seen=set()
    seen.add(stp)
    out.append(real_id[stp])
    while (len(stack)>0):
        vertex=stack.pop()
        nodes=children[vertex]
        for w in nodes:
            if w not in seen:
               out.append(real_id[w])
               stack.append(w)
               seen.add(w)
    return out

def split_v2(points,soma):
    long_path=0
    real_id=points.index.values
    points=points.reset_index(drop=True)
    children=child_parent(points)
    idlist=[]
    for i in range(len(points)):
        idlist.append(points['id'][i])     
    distance=[0 for i in range(len(points))]
    soma_id=soma.index.values[0]
    stack=[]
    stack.append(soma_id)
    seen=set()
    seen.add(soma_id)
    distance[soma_id]=0
    while (len(stack)>0):
        vertex=stack.pop()
        nodes=children[vertex]
        for w in nodes:
            if w not in seen:
                distance[w]=distance[vertex]+compute_dis(points,vertex,w)
                stack.append(w)
                seen.add(w)
    d_index={}
    for i in range(len(distance)):
        d_index[distance[i]]=i
    distance.sort(reverse=True)
    print("longest")
    longest_id=d_index[distance[0]]
    print(longest_id)
    print(distance[0])
    return real_id[longest_id]    
    

def define_dendrite(den,apic,label,soma):
    new_label=label
    for i in range(len(den)):
        new_label[den[i]]=soma
    if len(apic)>0:
        for i in range(len(apic)):
            # to change color or not
            new_label[apic[i]]=soma
    return new_label
                
def merge_dendritic_arbors(labels,df):
    L=[]
    new_labels=labels
    #Soma
    soma_df=df[df['type']==1]
    if len(soma_df)==0:
       soma_df=df[df['pid']==-1]    
    print("soma line")

    N=len(np.unique(labels))
    label_group=np.unique(labels)
    line_id=soma_df.index.values[0]
    print(line_id)

    d_label=labels[line_id]
    #print(d_label)
    
    
    # set dendritic range: child group of soma<- select the smallest cluster as the standard
    # dendrite based on type=3/4
    dendrite=df[df['type']==3]
    apic_d=df[df['type']==4]
    dendrite_id=dendrite.index.values
    apic_id=apic_d.index.values
    dendrite_avgd=avg_path(dendrite,soma_df)
    
    
    # dict: label-avg_distance pair
    label_avgd={}
    max_d=0
    min_d=1000

    for i in range(N):
        group_tmp=df[labels==label_group[i]]
        label_avgd[label_group[i]]=avg_path(group_tmp,soma_df)

    # 1. children of soma
    children=df[df['pid']==soma_df['id'][0]]
    cids=children.index.values
    children=children.reset_index(drop=True)
    print("children")
    print(len(children))
    cL=[]
    max_id=0
    if len(children)>0 :
        for i in range(len(children)):
            cL.append(labels[cids[i]])
            group_tmp=df[labels==labels[cids[i]]]
            avg_tmp=label_avgd[labels[cids[i]]]
            print("each cluster")
            print(labels[cids[i]])
            #print(avg_tmp)
            if avg_tmp>max_d:
                max_d=avg_tmp
                max_id=i
            if avg_tmp<min_d:
                min_d=avg_tmp

    print("soma color")
    print(d_label)

    for i in range(len(children)):
        if labels[cids[i]] != d_label:
            if max_d > 2*dendrite_avgd:
                max_d=2*dendrite_avgd
                continue
            L.append(labels[cids[i]])
    print(len(L))
    
    # 2. close groups
    for i in range(N):
        #print("distance")
        if label_group[i] not in L:
            print("check cluster")
            print(label_group[i])
            group_tmp=df[labels==label_group[i]]
            close_d=compute_shortest_path(group_tmp,soma_df)

            # combine close groups
            if (label_avgd[label_group[i]]<max_d) and (close_d<min_d):
                L.append(label_group[i])
                
    unique_L=np.unique(L) 
    print("change label")
    print(len(unique_L))
    print(unique_L)   
    
    # split one-color neuron
    print("split")
    print(N)
    print(len(unique_L))
   
    if N-len(unique_L)-1<1:
        print("one-color")
        twolabels=labels
        #axon=split(df,soma_df)
        #print(len(df))
        #print(len(axon))
        #for i in range(len(labels)):
        #    if i in axon:
        #        twolabels[i]=1
        #    else: 
        #        twolabels[i]=0
        #return twolabels        
        long_id=split_v2(df,soma_df)
        print(labels[long_id])
        axon=df[labels==labels[long_id]]
        #print(labels)
        axon_ids=axon.index.values
        for i in range(len(labels)):
            if i in axon_ids:
                twolabels[i]=1
            else:
                twolabels[i]=0
        return twolabels          
    else:
        if len(unique_L)>0:
            for i in range(len(unique_L)):
                for j in range(len(labels)):
                    if labels[j]==unique_L[i]:
                        new_labels[j]=d_label
        return new_labels


def aarbor_adaptive_spectral_swc(filename, min_my_n_clusters, max_my_n_clusters):
    
    generate_tmp_swc(filename)
    
    df = pandas.read_csv('./tmp_swc.csv',
                         sep=' ')

    X = df[['x', 'y', 'z']]
    Y = X.values

    print("read file")

    XS = np.zeros((len(X),len(X)))

#build the index table
    xyz3 = np.zeros((len(X),3))
    for i in range(len(X)):     
        cid = df['id'][i]-1
        xyz3[cid,:] = [Y[i,0], Y[i,1], Y[i,2]]    
#        XS[i,i] = 0.999;
    
    for i in range(len(X)):     

        cp = df['pid'][i]-1 #current parent id
#        print(cp)
        
        if cp<=0:
            continue;
        
        cid = df['id'][i]-1
        dx = (xyz3[i,0]-xyz3[cp,0])
        dy = (xyz3[i,1]-xyz3[cp,1])
        dz = (xyz3[i,2]-xyz3[cp,2])
        
        XS[cid, cp] = XS[cp, cid] = math.exp(-math.sqrt(dx*dx+dy*dy+dz*dz)/100)
    #    print(X)
    
    th = 0.0001    # ln(0.0001) = -9.2
    for i in range(len(X)):     
        for j in range(len(X)):     
            if XS[i][j] < th:
                dx = (xyz3[i,0]-xyz3[j,0])
                dy = (xyz3[i,1]-xyz3[j,1])
                dz = (xyz3[i,2]-xyz3[j,2])
                XS[i][j] = math.exp(-math.sqrt(dx*dx+dy*dy+dz*dz))
                if XS[i][j]>th:
                    XS[i][j] = th;
    print(XS)
    
    print("finish XS matrix")
    # #############################################################################
    # Compute clustering 
    
    min_score = -1;
    best_n_id = 0;
    my_labels = 0;
    
    for n in range(min_my_n_clusters, max_my_n_clusters+1):
        v = s_clustering(XS, Y, n)
        cur_score = v[0]
        cur_labels = v[1]
        print([n, cur_score])
        if min_score < 0:
            min_score = cur_score;
            best_n_id = n;
            my_labels = cur_labels;            
            continue
        
        if cur_score < min_score:
            min_score = cur_score;
            best_n_id = n;
            my_labels = cur_labels;            
    
    print(my_labels)

#    my_labels = pairwise_distances_argmin(X, my_cluster_centers)

    fout = open(filename + '.autoarbor_m3.arborstat.txt' , "wt")

    
    # #############################################################################
    # Plot result
    
    fig = plt.figure(figsize=(8, 8)) #fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#FFFF00', '#00FFFF', '#F0F0F0', '#0F0F0F', '#F0FF00', '#FFF000']
    
    
    
    # KMeans
#    ax = fig.add_subplot(1, 3, 1)

    np.set_printoptions(precision=2)
    
    line = 'arbor_id arbor_node_count arbor_center_x arbor_center_y arbor_center_z\n'
    fout.write( line )
    
    ax = fig.add_subplot(1, 1, 1)
    for k, col in zip(range(best_n_id), colors):
        print(k)
        my_members = my_labels == k
        cluster_center = np.mean(Y[my_members, :], axis=0, dtype=np.float64) 
        cluster_std = np.std(Y[my_members, :], axis=0, dtype=np.float64)
        print(cluster_std)
                
        ax.plot(Y[my_members, 0], Y[my_members, 1], 'w',
                markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6)
        
        line = str(k+1) + ' ' + str(np.count_nonzero(my_members)) + ' ' + str(cluster_center)[1:-1] + '\n'
        #indeed need to calculate the arbor length in the future, 
        # not just the count of nodes
    
        fout.write( line )

    # revise clustering
    new_labels=merge_dendritic_arbors(my_labels,df)
    #print("new label")
    #print(new_labels)

    df['labels'] = new_labels #add a column  
    df.columns=['#id','type','x','y','z','r','pid','labels']
    df.to_csv(filename + '._m3_l.eswc', index=False, sep=' ')
    df['type'] = new_labels #add a column  
    df.to_csv(filename + '._m3_lt.eswc', index=False, sep=' ')

    ax.set_title('AutoArbor_M3') #    ax.set_title('KMeans')
    ax.set_xticks(())
    ax.set_yticks(())
#    plt.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (
#        t_batch, k_means.inertia_))
    
#    plt.show()
    
    plt.savefig( filename + '.autoarbor_m3.pdf' )
    plt.close()
    
    if os.path.exists(r'./tmp_swc.csv'):
        os.remove(r'./tmp_swc.csv')

    fout.close()


# Main program starts here

parser = argparse.ArgumentParser()
parser.add_argument('--filename', help='SWC file name', type=str)
args = parser.parse_args()

aarbor_adaptive_spectral_swc(args.filename, 2, 4)



