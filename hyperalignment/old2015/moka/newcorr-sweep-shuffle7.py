# Using the Monge-Kantorovitch solution to hyperalign fMRI data



"""
THIS ONE DOES IT FOR SHUFFLED OF 7!

"""

import moka
import csv
import numpy
import scipy
import scipy.stats
import scipy.linalg
import scipy.weave
import xifti as xi
import copy
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, FastICA
from sklearn.metrics import confusion_matrix


from numba import autojit,jit,double,uint32

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

#tc = [copy.deepcopy(ds_all[0]) for ds in ds_all[1:]]
tc = copy.deepcopy(ds_all[0])
sc = [copy.deepcopy(ds) for ds in ds_all[1:]]

#####################TOP STUFF HERE####################################################

print "Top Voxels"
#topV_t = []
topV_t = moka.fast_top_voxels(tc['data'], sc[0]['data'], len(tc['data']))
tc['data'] = tc['data'][topV_t]

topV_s = []

for i in range(0, len(sc)):
	#topV_t.append(moka.fast_top_voxels(tc[i]['data'], sc[i]['data'], len(tc[i]['data'])))
	topV_s.append(moka.fast_top_voxels(sc[i]['data'], tc['data'], len(sc[i]['data'])))

csvout = csv.writer(open('newcorr-sweep-shuffle7-stz.csv', 'w'))

#num_voxels = 100
#3rd and 4th to 500 and 150
for TV in range(40, 1501, 10):

	num_voxels = TV
	tcc = copy.deepcopy(tc)
	scc = [copy.deepcopy(X)	for X in sc]
	
	tcc['data'] = tc['data'][topV_t[-num_voxels:]]
	for i in range(0, len(sc)):
		scc[i]['data'] = sc[i]['data'][topV_s[i][-num_voxels:]]

	#####################PCA STUFF HERE####################################################

	for PC in range(2, 20, 1):
		t = copy.deepcopy(tcc)
		s = [copy.deepcopy(X) for X in scc]
		print "\n\nPCA"
		num_pcs = PC
		pca = PCA(num_pcs)
		sData = []

		print str(TV) + "\t" + str(PC)

		pca.fit(t['data'].transpose())
		t['data'] = pca.transform(t['data'].transpose()).transpose()
		tData = t['data'].transpose()

		for i in range(0, len(s)):

			pca.fit(s[i]['data'].transpose())
			s[i]['data'] = pca.transform(s[i]['data'].transpose()).transpose()
			sData.append(s[i]['data'].transpose())

		#####################PART STUFF HERE###################################################	

		score1 = []
		score2 = []

		block = 7			##CHANGE HOW MANY TO TEST

		for part in range(0, len(labels), block): #I thought I could make this 1... apparently no

			#tDataTrain = []
			sDataTrain = []
			#tDataGen = []
			sDataGen = []
			
			tDataTrain = tData[part:part+block]
			tDataGen = numpy.concatenate((tData[:part], tData[part+block:]))

			tlabels = labels[part:part+block]

			for i in range(0, len(s)):

				sDataTrain.append(sData[i][part:part+block])
				sDataGen.append(numpy.concatenate((sData[i][:part], sData[i][part+block:])))

			#####################MAPP STUFF HERE###################################################

			new_s = []
			new_sData = []

			t['data'] = tDataGen.transpose()
			for i in range(0, len(s)):

				s[i]['data'] = sDataGen[i].transpose()

				T,b = moka.build_xform(s[i],t)

				s[i]['data'] = sDataTrain[i].transpose()

				new_s.append(moka.apply_xform(s[i],T,b))

				new_sData.append(new_s[i]['data'].transpose())

			#####################PREP STUFF HERE###################################################

			SA = sDataTrain[0]
			FA = new_sData[0]

			TRAIN = []
			TRAIN.append(tDataTrain)
			LABELS = []
			LABELS.append(tlabels)
			for i in range(1, len(s)):
				TRAIN.append(new_sData[i])
				LABELS.append(tlabels)

			TRAIN = numpy.concatenate(TRAIN)
			LABELS = numpy.concatenate(LABELS)	
			#####################CLASS STUFF HERE##################################################

		
			pig1 = svm.SVC(kernel='linear')
			pig2 = KNeighborsClassifier(1)

			D_train, D_test, L_train, L_test = cross_validation.train_test_split(TRAIN, LABELS, test_size = (float(30)/100.0), random_state=0)

			pig1.fit(D_train, L_train)
			pig2.fit(D_train, L_train)

			#print "SVM-Train"
			#print pig1.score(D_train, L_train)
			#print "KNN-Train"
			#print pig2.score(D_train, L_train)
			#print "\n"

			#print "SVM-Test"
			#print pig1.score(D_test, L_test)
			#print "KNN-Test"
			#print pig2.score(D_test, L_test)
			#print "\n"

			#print "SVM-SA"
			#print pig1.score(SA, tlabels)
			#print "KNN-SA"
			#print pig2.score(SA, tlabels)
			#print "\n"

			#print "SVM-FA"
			score1.append(pig1.score(FA, tlabels))
			#print score1
			#print "KNN-FA"
			score2.append(pig2.score(FA, tlabels))
			#print score2
			#print "\n"


		#print "SVM-FA"
		#print score1
		print numpy.mean(score1)
		#print "KNN-FA"
		#print score2
		print numpy.mean(score2)
		print "\n"
		csvout.writerow([TV, PC, numpy.mean(score1), numpy.mean(score2)])
