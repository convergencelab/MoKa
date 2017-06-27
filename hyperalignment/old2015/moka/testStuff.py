import csv
import numpy
import scipy
import xifti as xi
import moka
import copy
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, FastICA
from sklearn.metrics import confusion_matrix

from mvpa2.suite import *

print "Loading Data"
filepath = "/home/james/Dropbox/SchoolStuff/UWO/FunctionalAlignment/hyperalignment/datadb/hyperalignment_tutorial_data_2.4/hyperalignment_tutorial_data_2.4.hdf5.gz"

ds_all_IN = h5load(filepath)

labels =  ds_all_IN[0].targets

ds_all = [xi.pack_vtx(numpy.array(ds.samples.transpose()),'null','VTS','Voxel Time Series [channels,volumes]','null','null','null') for ds in ds_all_IN]
ds_all = numpy.array(ds_all)
swap_ds = ds_all[6]			#because 6 seems to work soo well and this will make indexing easier
ds_all[6] = ds_all[0]
ds_all[0] = swap_ds

ds_all = ds_all[0:2]

tc = [copy.deepcopy(ds_all[0]) for ds in ds_all[1:]]
sc = [copy.deepcopy(ds) for ds in ds_all[1:]]

#####################TOP STUFF HERE####################################################

print "Top Voxels"
topV_t = []
topV_s = []
for i in range(0, len(sc)):
	topV_t.append(moka.fast_top_voxels(tc[i]['data'], sc[i]['data'], len(tc[i]['data'])))
	topV_s.append(moka.fast_top_voxels(sc[i]['data'], tc[i]['data'], len(sc[i]['data'])))


print tc[0]['data'][0]

tcc = [copy.deepcopy(X) for X in tc]
scc = [copy.deepcopy(X)	for X in sc]
for i in range(0, len(sc)):
	tcc[i]['data'] = tc[i]['data'][topV_t[i][-100:]]
	scc[i]['data'] = sc[i]['data'][topV_s[i][-100:]]
print
print
print tcc[0]['data'][0]
print
print

moka.fast_top_voxels(scc[i]['data'], tc[i]['data'], len(scc[i]['data']))
