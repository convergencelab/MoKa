# Using the Monge-Kantorovitch solution to hyperalign fMRI data
####NEED THIS export PYTHONPATH=$PWD


"""
THIS ONE DOES IT FOR SHUFFLED OF 7!
THIS ONE ALSO CHANGES THE ONE BEING TESTED IN THE MAPPING

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


##########################################
from mvpa2.suite import *


print "Loading Data"
filepath = "/home/james/Dropbox/SchoolStuff/UWO/FunctionalAlignment/hyperalignment/datadb/hyperalignment_tutorial_data_2.4/hyperalignment_tutorial_data_2.4.hdf5.gz"

ds_all_IN = h5load(filepath)

labels =  ds_all_IN[0].targets

ds_all = [xi.pack_vtx(numpy.array(ds.samples.transpose()),'null','VTS','Voxel Time Series [channels,volumes]','null','null','null') for ds in ds_all_IN]
ds_all = numpy.array(ds_all)
swap_ds = ds_all[6]			#6 seems to work soo well and this will make indexing easier
ds_all[6] = ds_all[0]
ds_all[0] = swap_ds

#swap_ds = ds_all[7]			# test other Source
#ds_all[7] = ds_all[1]
#ds_all[1] = swap_ds

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

csvout = csv.writer(open('big0-stz.csv', 'w'))

#num_voxels = 100
#start at 40
for TV in range(40, 120, 1):

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

		
		for i in range(0, len(s)):
			pca.fit(t[i]['data'].transpose())
			t[i]['data'] = pca.transform(t[i]['data'].transpose()).transpose()

			pca.fit(s[i]['data'].transpose())
			s[i]['data'] = pca.transform(s[i]['data'].transpose()).transpose()

			tData.append(t[i]['data'].transpose())
			sData.append(s[i]['data'].transpose())


		#####################HOLD STUFF HERE###################################################	

		for holdOut in range(30, 85, 5):
			
			print str(TV) + "\t" + str(PC) + "\t" + str(holdOut)

			#####################PART STUFF HERE###################################################	
		
			score1 = []
			score2 = []
			score3 = []
			score4 = []
			score5 = []
			score6 = []
			score7 = []

			for part in range(0, len(labels), 7): #I thought I could make this 1... apparently no
			
				tDataTrain = []
				sDataTrain = []
				tDataGen = []
				sDataGen = []
			
			

				tlabels = labels[part:part+7]
				for i in range(0, len(s)):
					tDataTrain.append(tData[i][part:part+7])
					tDataGen.append(numpy.concatenate((tData[i][:part], tData[i][part+7:])))

					sDataTrain.append(sData[i][part:part+7])
					sDataGen.append(numpy.concatenate((sData[i][:part], sData[i][part+7:])))


		


				#####################MAPP STUFF HERE###################################################

				new_s = []
				new_sData = []

				for i in range(0, len(s)):
					t[i]['data'] = tDataGen[i].transpose()
					s[i]['data'] = sDataGen[i].transpose()

					T,b = moka.build_xform(s[i],t[i])

					s[i]['data'] = sDataTrain[i].transpose()

					new_s.append(moka.apply_xform(s[i],T,b))

					new_sData.append(new_s[i]['data'].transpose())

				#####################PREP STUFF HERE###################################################

				SA = sDataTrain[0]
				FA = new_sData[0]

				TRAIN = []
				TRAIN.append(tDataTrain[0])
				LABELS = []
				LABELS.append(tlabels)
				for i in range(1, len(s)):
					TRAIN.append(tDataTrain[i])		
					LABELS.append(tlabels)						
					TRAIN.append(new_sData[i])
					LABELS.append(tlabels)

				TRAIN = numpy.concatenate(TRAIN)
				LABELS = numpy.concatenate(LABELS)	
	
				#####################CLASS STUFF HERE##################################################

	
				pig1 = svm.SVC(kernel='linear')
				pig2 = KNeighborsClassifier(n_neighbors=1)
				pig3 = KNeighborsClassifier(n_neighbors=2)
				pig4 = KNeighborsClassifier(n_neighbors=3)
				pig5 = KNeighborsClassifier(n_neighbors=4)
				pig6 = KNeighborsClassifier(n_neighbors=5)
				pig7 = svm.LinearSVC()

				D_train, D_test, L_train, L_test = cross_validation.train_test_split(TRAIN, LABELS, test_size = (float(holdOut)/100.0), random_state=0)

				pig1.fit(D_train, L_train)
				pig2.fit(D_train, L_train)
				pig3.fit(D_train, L_train)
				pig4.fit(D_train, L_train)
				pig5.fit(D_train, L_train)
				pig6.fit(D_train, L_train)
				pig7.fit(D_train, L_train)

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
				score3.append(pig2.score(FA, tlabels))
				score4.append(pig2.score(FA, tlabels))
				score5.append(pig2.score(FA, tlabels))
				score6.append(pig2.score(FA, tlabels))
				score7.append(pig2.score(FA, tlabels))

			print "1"
			print score1
			print numpy.mean(score1)
			print "2"
			print score2
			print numpy.mean(score2)
			print "3"
			print score3
			print numpy.mean(score3)
			print "4"
			print score4
			print numpy.mean(score4)
			print "5"
			print score5
			print numpy.mean(score5)
			print "6"
			print score6
			print numpy.mean(score6)
			print "7"
			print score7
			print numpy.mean(score7)
			print "\n"
			csvout.writerow([TV, PC, holdOut, numpy.mean(score1), numpy.mean(score2), numpy.mean(score3), numpy.mean(score4), numpy.mean(score5), numpy.mean(score6), numpy.mean(score7)])
