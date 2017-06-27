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

t1 = copy.deepcopy(ds_all[6])
s1 = copy.deepcopy(ds_all[1])
t2 = copy.deepcopy(ds_all[6])
s2 = copy.deepcopy(ds_all[2])

#####################TOP STUFF HERE####################################################

print "Top Voxels"
topV_t1 = fast_top_voxels(t1['data'], s1['data'], len(t1['data']))
topV_s1 = fast_top_voxels(s1['data'], t1['data'], len(s1['data']))
topV_t2 = fast_top_voxels(t2['data'], s2['data'], len(t2['data']))
topV_s2 = fast_top_voxels(s2['data'], t2['data'], len(s2['data']))

num_voxels = 100
t1['data'] = t1['data'][topV_t1[-100:]]
s1['data'] = s1['data'][topV_s1[-100:]]
t2['data'] = t2['data'][topV_t2[-850:]]
s2['data'] = s2['data'][topV_s2[-850:]]

#####################PCA STUFF HERE####################################################

print "\n\nPCA\n"
num_pcs = 7

pca = PCA(num_pcs)
pca.fit(t1['data'].transpose())
t1['data'] = pca.transform(t1['data'].transpose()).transpose()
print "Explained Variance T1:"
print numpy.sum(pca.explained_variance_ratio_)

pca.fit(s1['data'].transpose())
s1['data'] = pca.transform(s1['data'].transpose()).transpose()
print "Explained Variance S1:"
print numpy.sum(pca.explained_variance_ratio_)

pca.fit(t2['data'].transpose())
t2['data'] = pca.transform(t2['data'].transpose()).transpose()
print "Explained Variance T2:"
print numpy.sum(pca.explained_variance_ratio_)

pca.fit(s2['data'].transpose())
s2['data'] = pca.transform(s2['data'].transpose()).transpose()
print "Explained Variance S2:"
print numpy.sum(pca.explained_variance_ratio_)

t1Data = t1['data'].transpose()
s1Data = s1['data'].transpose()
t2Data = t2['data'].transpose()
s2Data = s2['data'].transpose()



t1DataTrain = t1Data[0:21]
t1DataGen = t1Data[21:]

s1DataTrain = s1Data[0:21]
s1DataGen = s1Data[21:]

t2DataTrain = t2Data[0:21]
t2DataGen = t2Data[21:]

s2DataTrain = s2Data[0:21]
s2DataGen = s2Data[21:]

labels = labels[0:21]


#####################MAPP STUFF HERE###################################################


t1['data'] = t1DataGen.transpose()
s1['data'] = s1DataGen.transpose()

T,b = build_xform(s1,t1)

s1['data'] = s1DataTrain.transpose()

new_s1 = apply_xform(s1,T,b)

new_s1Data = new_s1['data'].transpose()

t2['data'] = t2DataGen.transpose()
s2['data'] = s2DataGen.transpose()

T,b = build_xform(s2,t2)

s2['data'] = s2DataTrain.transpose()

new_s2 = apply_xform(s2,T,b)

new_s2Data = new_s2['data'].transpose()

tDataTrain = numpy.concatenate((t1DataTrain, new_s2Data))
labels2 = numpy.concatenate((labels, labels))





#####################CLASS STUFF HERE##################################################

print "\n\n"
pig1 = svm.SVC(kernel='linear')
pig2 = KNeighborsClassifier(1)

D_train, D_test, L_train, L_test = cross_validation.train_test_split(tDataTrain, labels2, test_size = (float(30)/100.0), random_state=0)

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
print pig1.score(s1DataTrain, labels)
print "KNN-SA"
print pig2.score(s1DataTrain, labels)
print "\n"

print "SVM-FA"
print pig1.score(new_s1Data, labels)
print "KNN-FA"
print pig2.score(new_s1Data, labels)
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


pred_SVM = pig1.predict(new_s1Data);
pred_KNN = pig2.predict(new_s1Data);

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

