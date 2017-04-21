"""
Copyright Elliot Schumacher
Created 4/21/17
"""

from sklearn.manifold import TSNE
import numpy as np
import os
import pickle
import sys

def main():
    vect_directory = sys.argv[1]#"/Users/elliotschumacher/Dropbox/git/Semantics-Final-Project/vectors"
    orig_vectors_file = "vector_file.txt.sense"
    retro_vects_file = "en-zh_retrofit_vectors.pkl"
    contx_vects_file = "en-zh_retrofit_context_vectors.pkl"
    orig_vects_dict = {}
    retro_vects_dict = {}
    con_vects_dict = {}
    vects_combined = np.zeros(shape=(300,))
    with open(os.path.join(vect_directory, contx_vects_file), 'rb') as pickle_file:
        con_vects_dict = pickle.load(pickle_file, encoding='latin1')


    print("Loaded cont")
    with open(os.path.join(vect_directory, retro_vects_file), 'rb') as pickle_file:
        retro_vects_dict = pickle.load(pickle_file,  encoding='latin1')
        for c in retro_vects_dict:
            print(c)
    print("Loaded retro")
    print(len(retro_vects_dict))
    print(len(con_vects_dict))


    with open(os.path.join(vect_directory, orig_vectors_file)) as vector_file:
        first_line = True
        for l in vector_file:
            if first_line:
                first_line = False
            else:
                try:
                    space_delim = l.split()
                    word = space_delim[0].split('|')[0]
                    sense = space_delim[0].split('|')[1].strip(':')
                    vects_combined = np.concatenate(vects_combined, np.asarray(space_delim[1:]), axis=0)
                    orig_vects_dict[(word, sense)] = np.asarray(space_delim[1:])

                except:
                    print(l)
    print("Loaded orig")


    pass


    pass

if __name__ == '__main__': main()
