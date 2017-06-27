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

#ds_all = ds_all[0:3]

tc = [copy.deepcopy(ds_all[0]) for ds in ds_all[1:]]
sc = [copy.deepcopy(ds) for ds in ds_all[1:]]

#####################TOP STUFF HERE####################################################

print "Top Voxels"
topV_t = []
topV_s = []
for i in range(0, len(sc)):
	topV_t.append(moka.fast_top_voxels(tc[i]['data'], sc[i]['data'], len(tc[i]['data'])))
	topV_s.append(moka.fast_top_voxels(sc[i]['data'], tc[i]['data'], len(sc[i]['data'])))

csvout = csv.writer(open('nohold-sweep-stz.csv', 'w'))

#num_voxels = 100
#3rd and 4th to 500 and 150
for TV in range(40, 1501, 1):

	num_voxels = TV
	tcc = [copy.deepcopy(X) for X in tc]
	scc = [copy.deepcopy(X)	for X in sc]
	for i in range(0, len(sc)):
		tcc[i]['data'] = tc[i]['data'][topV_t[i][-num_voxels:]]
		scc[i]['data'] = sc[i]['data'][topV_s[i][-num_voxels:]]

	#####################PCA STUFF HERE####################################################

	for PC in range(2, 25, 1):
		t = [copy.deepcopy(X) for X in tcc]
		s = [copy.deepcopy(X) for X in scc]
		print "\n\nPCA"
		num_pcs = PC
		pca = PCA(num_pcs)
		tData = []
		sData = []
		tDataTrain = []
		sDataTrain = []
		tDataGen = []
		sDataGen = []

		for i in range(0, len(s)):
			pca.fit(t[i]['data'].transpose())
			t[i]['data'] = pca.transform(t[i]['data'].transpose()).transpose()

			pca.fit(s[i]['data'].transpose())
			s[i]['data'] = pca.transform(s[i]['data'].transpose()).transpose()

			tData.append(t[i]['data'].transpose())
			sData.append(s[i]['data'].transpose())

			#tDataTrain.append(tData[i][0:7])		#WAS 21! (SHUFFLE THESE AROUND)
			#tDataGen.append(tData[i][7:])

			#sDataTrain.append(sData[i][0:7])
			#sDataGen.append(sData[i][7:])

		#labels = labels[0:7]

		#####################MAPP STUFF HERE###################################################

		new_s = []
		new_sData = []

		for i in range(0, len(s)):
			t[i]['data'] = tData[i].transpose()
			s[i]['data'] = sData[i].transpose()

			T,b = moka.build_xform(s[i],t[i])

			#s[i]['data'] = sData[i].transpose()

			new_s.append(moka.apply_xform(s[i],T,b))

			new_sData.append(new_s[i]['data'].transpose())

		#####################PREP STUFF HERE###################################################

		SA = sData[0]
		FA = new_sData[0]		#0 is being tested here

		TRAIN = []
		TRAIN.append(tData[0])
		LABELS = []
		LABELS.append(labels)
		for i in range(1, len(s)):		#train on everyone else but 0
			TRAIN.append(tData[i])		####JUST TRYING THIS...let's see...
			LABELS.append(labels)		####
			TRAIN.append(new_sData[i])
			LABELS.append(labels)

		TRAIN = numpy.concatenate(TRAIN)
		LABELS = numpy.concatenate(LABELS)	
		#####################CLASS STUFF HERE##################################################

		print "\n"
		pig1 = svm.SVC(kernel='linear')
		pig2 = KNeighborsClassifier(1)

		D_train, D_test, L_train, L_test = cross_validation.train_test_split(TRAIN, LABELS, test_size = (float(30)/100.0), random_state=0)

		pig1.fit(D_train, L_train)
		pig2.fit(D_train, L_train)

		print "SVM-Train"
		print pig1.score(D_train, L_train)
		print "KNN-Train"
		print pig2.score(D_train, L_train)
		print "\n"

		print "SVM-Test"
		print pig1.score(D_test, L_test)
		print "KNN-Test"
		print pig2.score(D_test, L_test)
		print "\n"

		print "SVM-SA"
		print pig1.score(SA, labels)
		print "KNN-SA"
		print pig2.score(SA, labels)
		print "\n"

		print "SVM-FA"
		score1 = pig1.score(FA, labels)
		print score1
		print "KNN-FA"
		score2 = pig2.score(FA, labels)
		print score2
		print "\n"

		csvout.writerow([TV, PC, score1, score2])
