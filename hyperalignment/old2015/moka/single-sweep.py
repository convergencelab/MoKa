import csv
import numpy
import scipy
import scipy.stats
import scipy.linalg
import scipy.weave
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
from mvpa2.support.pylab import pl
from mvpa2.misc.data_generators import noisy_2d_fx
from mvpa2.mappers.svd import SVDMapper
from mvpa2.mappers.mdp_adaptor import ICAMapper, PCAMapper
from mvpa2 import cfg


print "Loading Data"
#filepath = os.path.join(pymvpa_datadbroot, '../HaxbyHAL/hyperalignment_tutorial_data_2.4', "hyperalignment_tutorial_data_2.4.hdf5.gz")
filepath = "/home/james/Dropbox/SchoolStuff/UWO/FunctionalAlignment/hyperalignment/datadb/hyperalignment_tutorial_data_2.4/hyperalignment_tutorial_data_2.4.hdf5.gz"

ds_all_IN = h5load(filepath)

labels =  ds_all_IN[0].targets

ds_all = [xi.pack_vtx(numpy.array(ds.samples.transpose()),'null','VTS','Voxel Time Series [channels,volumes]','null','null','null') for ds in ds_all_IN]
ds_all = numpy.array(ds_all)

tc = copy.deepcopy(ds_all[6])
sc = copy.deepcopy(ds_all[1])

#####################TOP STUFF HERE####################################################

print "Top Voxels"
topV_t = moka.fast_top_voxels(tc['data'], sc['data'], len(tc['data']))
topV_s = moka.fast_top_voxels(sc['data'], tc['data'], len(sc['data']))

csvout = csv.writer(open('stz.csv', 'w'))

for TV in range(40, 1000, 1):

	num_voxels = TV
	tcc = copy.deepcopy(tc)
	scc = copy.deepcopy(sc)	
	tcc['data'] = tc['data'][topV_t[-num_voxels:]]
	scc['data'] = sc['data'][topV_s[-num_voxels:]]

	#####################PCA STUFF HERE####################################################
	for PC in range(2, 25, 1):
		t = copy.deepcopy(tcc)
		s = copy.deepcopy(scc)	
		print "\n\nPCA\n"
		num_pcs = PC
		pca = PCA(num_pcs)
		pca.fit(tcc['data'].transpose())
		t['data'] = pca.transform(tcc['data'].transpose()).transpose()
		#print "Explained Variance T:"
		#print numpy.sum(pca.explained_variance_ratio_)

		pca.fit(scc['data'].transpose())
		s['data'] = pca.transform(scc['data'].transpose()).transpose()
		#print "Explained Variance S:"
		#print numpy.sum(pca.explained_variance_ratio_)

		tData = t['data'].transpose()
		sData = s['data'].transpose()

		tDataTrain = tData[0:7]
		tDataGen = tData[7:]

		sDataTrain = sData[0:7]
		sDataGen = sData[7:]

		labels = labels[0:7]


		#####################MAPP STUFF HERE###################################################


		t['data'] = tDataGen.transpose()
		s['data'] = sDataGen.transpose()

		T,b = moka.build_xform(s,t)

		s['data'] = sDataTrain.transpose()

		new_s = moka.apply_xform(s,T,b)

		new_sData = new_s['data'].transpose()

		#####################CLASS STUFF HERE##################################################

		print "\n\n"
		pig1 = svm.SVC(kernel='linear')
		pig2 = KNeighborsClassifier(1)

		D_train, D_test, L_train, L_test = cross_validation.train_test_split(tDataTrain, labels, test_size = (float(30)/100.0), random_state=0)

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
		print pig1.score(sDataTrain, labels)
		print "KNN-SA"
		print pig2.score(sDataTrain, labels)
		print "\n"

		print "SVM-FA"
		score1 = pig1.score(new_sData, labels)
		print score1
		print "KNN-FA"
		score2 = pig2.score(new_sData, labels)
		print score2
		print "\n"

		csvout.writerow([TV, PC, score1, score2])
