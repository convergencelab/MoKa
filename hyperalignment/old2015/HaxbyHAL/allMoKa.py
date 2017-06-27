# Using the Monge-Kantorovitch solution to hyperalign fMRI data



"""
Voxels from the VT mask of each subject were partitioned into left and right hemisphere voxels. 
For each voxel in a subject, the correlation of its time-series with the time-series of each voxel 
in the same hemisphere of each other subject was calculated. 
The highest among these correlations was considered as that voxel's correlation-score. 
The sum of the correlation-scores for a voxel over all twenty subjects was considered its total-correlation-score. 
For each subject, we then ranked voxels in each hemisphere based on their total-correlation-scores 
(voxels with highest scores ranked the best). These ranks formed the basis for selecting a certain 
number of voxels from each subject's left and right hemispheres.

The other alternative is PCA, of course.
"""

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

# Build affine transformation matrices
#@jit(argtypes=[double[:], double[:,:], double[:,:]])
def build_xform(svtx, tvtx):
	"""
	Build a linear transform T,b that maps from the source vtx svtx onto the target vtx
	"""
	
	#print 'Building transform'
	source = svtx['data']
	target = tvtx['data']

	# Estimate means
	mu_s = numpy.mean(source,axis=1)
	mu_t = numpy.mean(target,axis=1)

	b = [mu_s,mu_t]

	# Generate covariance matrices
	s_s = numpy.cov(source)
	s_t = numpy.cov(target)


	# Generate the transformation matrix via the Monge-Kantorovitch equation!
	T = numpy.linalg.inv(numpy.real(scipy.linalg.sqrtm(s_s))).dot( numpy.real(scipy.linalg.sqrtm(( numpy.real(scipy.linalg.sqrtm(s_s))).dot(s_t).dot(numpy.real(scipy.linalg.sqrtm(s_s)))))).dot(numpy.linalg.inv(numpy.real(scipy.linalg.sqrtm(s_s))))

	return T,b

def build_xform_gauss(svtx, tvtx):
	"""
	Build a linear transform T,b that maps from the source vtx svtx onto the target vtx with gaussian noise added to the matrix to eliminate singularity issues.
	"""
	
	#print 'Building gaussian transform'
	source = svtx['data']
	target = tvtx['data']
	
		
	for i in range(len(source)):
		for j in range(len(source[0])):
			source[i][j] = source[i][j] + numpy.random.normal(0, 0.00001)

	# Estimate means
	mu_s = numpy.mean(source,axis=1)
	mu_t = numpy.mean(target,axis=1)

	b = [mu_s,mu_t]

	# Generate covariance matrices
	s_s = numpy.cov(source)
	s_t = numpy.cov(target)


	# Generate the transformation matrix via the Monge-Kantorovitch equation!
	T = numpy.linalg.inv(numpy.real(scipy.linalg.sqrtm(s_s))).dot( numpy.real(scipy.linalg.sqrtm(( numpy.real(scipy.linalg.sqrtm(s_s))).dot(s_t).dot(numpy.real(scipy.linalg.sqrtm(s_s)))))).dot(numpy.linalg.inv(numpy.real(scipy.linalg.sqrtm(s_s))))

	return T,b

def apply_xform(vtx,T,b):
	"""
	Given Monge-Kantrovich linear transformation defined by T,b; map the given vtx into
	the space defined by the transform.
	"""

	#print 'Applying transform'
	outmat = numpy.zeros(vtx['data'].shape)
	
	dat = vtx['data']
	
	for i,func_point in enumerate(dat.transpose()):	
		outmat[:,i] = T.dot(func_point-b[0]) + b[1]
	
	outvtx = copy.deepcopy(vtx)
	outvtx['data'] = outmat
	
	return outvtx

############
def build_avg_vtx(vtxs):
	"""
	Build an average vtx based on the array passed in. It returns this average VTX
	"""

	print 'Building Average VTX'
	outmat = numpy.zeros(vtxs[0]['data'].shape)
	avgvtx = copy.deepcopy(vtxs[0])
	
	for i in range(0, len(outmat)):
		for j in range(0, len(outmat[0])):
			for k in vtxs:
				outmat[i,j] = outmat[i,j] + k['data'][i,j]
			outmat[i,j] = outmat[i,j]/len(vtxs)


	avgvtx['data'] = outmat
	
	return avgvtx

############
def fast_top_voxels(xmat,ymat,topn=50):	
	code = 	'''
			int xp,yp,t;
			double x,y,xy,x2,y2,res;
			double maxscore;	

			for(xp=0;xp<n;xp++) {
	
			  maxscore=0;
			  printf("%i\\n",xp);
			  for(yp=0;yp<n;yp++) {
	  
				x=0;y=0;xy=0;x2=0;y2=0;
				
				for(t=0;t<m;t++){
					xy += xmat(xp,t) * ymat(yp,t);
					x +=  xmat(xp,t);
					y += ymat(yp,t);
					x2 += xmat(xp,t) * xmat(xp,t);
					y2 += ymat(yp,t) * ymat(yp,t);
					//printf("%f,%f\\n",mat(xp,t),mat(yp,t));	
				}
				
			    	

				res = (m*xy - x*y)/( sqrt(m*x2 - x*x)*sqrt(m*y2 - y*y));
				if (res > maxscore) maxscore = res;
	  
			 }
			 
			 score(xp) = maxscore;
			}
	
	
			'''

	score = numpy.zeros(xmat.shape[0])
	n = xmat.shape[0]
	m = xmat.shape[1]
	scipy.weave.inline(code,['xmat','ymat','n','m','score'],verbose = 1, compiler = 'gcc',  type_converters = scipy.weave.converters.blitz)
	indx = numpy.argsort(score)
	
	return indx[-topn:]



##########################################
##########################################
##########################################
##########################################
from mvpa2.suite import *
from mvpa2.support.pylab import pl
from mvpa2.misc.data_generators import noisy_2d_fx
from mvpa2.mappers.svd import SVDMapper
from mvpa2.mappers.mdp_adaptor import ICAMapper, PCAMapper
from mvpa2 import cfg


print "Loading Data"
filepath = os.path.join(pymvpa_datadbroot, 'hyperalignment_tutorial_data_2.4', "hyperalignment_tutorial_data_2.4.hdf5.gz")
#filepath = "/home/james/Dropbox/SchoolStuff/UWO/FunctionalAlignment/hyperalignment/HaxbyHAL/datadb/hyperalignment_tutorial_data_2.4/hyperalignment_tutorial_data_2.4.hdf5.gz"

ds_all_IN = h5load(filepath)

labels =  ds_all_IN[0].targets

ds_all = [xi.pack_vtx(numpy.array(ds.samples.transpose()),'null','VTS','Voxel Time Series [channels,volumes]','null','null','null') for ds in ds_all_IN]
ds_all = numpy.array(ds_all)
swap_ds = ds_all[6]			#because 6 seems to work soo well and this will make indexing easier
ds_all[6] = ds_all[0]
ds_all[0] = swap_ds

#ds_all = ds_all[0:6]

t = [copy.deepcopy(ds_all[0]) for ds in ds_all[1:]]
s = [copy.deepcopy(ds) for ds in ds_all[1:]]

#####################TOP STUFF HERE####################################################

print "Top Voxels"
topV_t = []
topV_s = []
for i in range(0, len(s)):
	topV_t.append(fast_top_voxels(t[i]['data'], s[i]['data'], len(t[i]['data'])))
	topV_s.append(fast_top_voxels(s[i]['data'], t[i]['data'], len(s[i]['data'])))


#num_voxels = 100
#3rd and 4th to 500 and 150
num_voxels = [100, 850, 150, 150, 750, 100, 400, 950, 100]
for i in range(0, len(s)):
	t[i]['data'] = t[i]['data'][topV_t[i][-num_voxels[i]:]]
	s[i]['data'] = s[i]['data'][topV_s[i][-num_voxels[i]:]]

#####################PCA STUFF HERE####################################################

print "\n\nPCA"
num_pcs = 14
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

	tDataTrain.append(tData[i][0:21])
	tDataGen.append(tData[i][21:])

	sDataTrain.append(sData[i][0:21])
	sDataGen.append(sData[i][21:])

labels = labels[0:21]


#####################MAPP STUFF HERE###################################################

new_s = []
new_sData = []

for i in range(0, len(s)):
	t[i]['data'] = tDataGen[i].transpose()
	s[i]['data'] = sDataGen[i].transpose()

	T,b = build_xform(s[i],t[i])

	s[i]['data'] = sDataTrain[i].transpose()

	new_s.append(apply_xform(s[i],T,b))

	new_sData.append(new_s[i]['data'].transpose())

#####################PREP STUFF HERE###################################################

SA = sDataTrain[0]
FA = new_sData[0]

TRAIN = []
TRAIN.append(tDataTrain[0])
LABELS = []
LABELS.append(labels)
for i in range(1, len(s)):
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
print pig1.score(FA, labels)
print "KNN-FA"
print pig2.score(FA, labels)
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

