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
#filepath = os.path.join(pymvpa_datadbroot, 'hyperalignment_tutorial_data_2.4', "hyperalignment_tutorial_data_2.4.hdf5.gz")
filepath = "/home/james/Dropbox/SchoolStuff/UWO/FunctionalAlignment/hyperalignment/datadb/hyperalignment_tutorial_data_2.4/hyperalignment_tutorial_data_2.4.hdf5.gz"

ds_all_IN = h5load(filepath)

labels =  ds_all_IN[0].targets

ds_all = [xi.pack_vtx(numpy.array(ds.samples.transpose()),'null','VTS','Voxel Time Series [channels,volumes]','null','null','null') for ds in ds_all_IN]
ds_all = numpy.array(ds_all)

t = copy.deepcopy(ds_all[6])
s = copy.deepcopy(ds_all[1])

#####################TOP STUFF HERE####################################################

print "Top Voxels"
topV_t = moka.fast_top_voxels(t['data'], s['data'], len(t['data']))
topV_s = moka.fast_top_voxels(s['data'], t['data'], len(s['data']))


num_voxels = 100
t['data'] = t['data'][topV_t[-num_voxels:]]
s['data'] = s['data'][topV_s[-num_voxels:]]

#####################PCA STUFF HERE####################################################

print "\n\nPCA\n"
num_pcs = 4

pca = PCA(num_pcs)
pca.fit(t['data'].transpose())
t['data'] = pca.transform(t['data'].transpose()).transpose()
print "Explained Variance T:"
print numpy.sum(pca.explained_variance_ratio_)

pca.fit(s['data'].transpose())
s['data'] = pca.transform(s['data'].transpose()).transpose()
print "Explained Variance S:"
print numpy.sum(pca.explained_variance_ratio_)

tData = t['data'].transpose()
sData = s['data'].transpose()

tDataTrain = tData[0:21]
tDataGen = tData[21:]

sDataTrain = sData[0:21]
sDataGen = sData[21:]

labels = labels[0:21]


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
print pig1.score(new_sData, labels)
print "KNN-FA"
print pig2.score(new_sData, labels)
print "\n"





'''
##################################
pred_SVM = pig1.predict(D_test);
pred_KNN = pig2.predict(D_test);

cm_SVM = confusion_matrix(L_test, pred_SVM)
cm_KNN = confusion_matrix(L_test, pred_KNN)

cm_normalized_SVM = cm_SVM.astype('float') / cm_SVM.sum(axis=1)[:, numpy.newaxis]
cm_normalized_KNN = cm_KNN.astype('float') / cm_KNN.sum(axis=1)[:, numpy.newaxis]

plt.figure()

cmap=plt.cm.Blues
plt.imshow(cm_normalized_SVM, interpolation='nearest', cmap=cmap)

for asd, cas in enumerate(cm_normalized_SVM):
	for sdf, c in enumerate(cas):
		if c>0:
			plt.text(sdf-.2, asd+.2, c, fontsize=14)

plt.title('SVM')
plt.colorbar(ticks=[0,1])
tick_marks = numpy.arange(len(numpy.unique(labels)))
plt.xticks(tick_marks, numpy.unique(labels), rotation=45)
plt.yticks(tick_marks, numpy.unique(labels))
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()

pred_SVM = pig1.predict(sData);
pred_KNN = pig2.predict(sData);

cm_SVM = confusion_matrix(labels, pred_SVM)
cm_KNN = confusion_matrix(labels, pred_KNN)

cm_normalized_SVM = cm_SVM.astype('float') / cm_SVM.sum(axis=1)[:, numpy.newaxis]
cm_normalized_KNN = cm_KNN.astype('float') / cm_KNN.sum(axis=1)[:, numpy.newaxis]

plt.figure()

cmap=plt.cm.Blues
plt.imshow(cm_normalized_SVM, interpolation='nearest', cmap=cmap)

for asd, cas in enumerate(cm_normalized_SVM):
	for sdf, c in enumerate(cas):
		if c>0:
			plt.text(sdf-.2, asd+.2, c, fontsize=14)

plt.title('SVM')
plt.colorbar(ticks=[0,1])
tick_marks = numpy.arange(len(numpy.unique(labels)))
plt.xticks(tick_marks, numpy.unique(labels), rotation=45)
plt.yticks(tick_marks, numpy.unique(labels))
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()


pred_SVM = pig1.predict(new_sData);
pred_KNN = pig2.predict(new_sData);

cm_SVM = confusion_matrix(labels, pred_SVM)
cm_KNN = confusion_matrix(labels, pred_KNN)

cm_normalized_SVM = cm_SVM.astype('float') / cm_SVM.sum(axis=1)[:, numpy.newaxis]
cm_normalized_KNN = cm_KNN.astype('float') / cm_KNN.sum(axis=1)[:, numpy.newaxis]

plt.figure()

cmap=plt.cm.Blues
plt.imshow(cm_normalized_SVM, interpolation='nearest', cmap=cmap)

for asd, cas in enumerate(cm_normalized_SVM):
	for sdf, c in enumerate(cas):
		if c>0:
			plt.text(sdf-.2, asd+.2, c, fontsize=14)

plt.title('SVM')
plt.colorbar(ticks=[0,1])
tick_marks = numpy.arange(len(numpy.unique(labels)))
plt.xticks(tick_marks, numpy.unique(labels), rotation=45)
plt.yticks(tick_marks, numpy.unique(labels))
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()

##################################
'''

