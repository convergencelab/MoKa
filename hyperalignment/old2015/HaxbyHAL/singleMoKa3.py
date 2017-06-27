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

tc = copy.deepcopy(ds_all[6])
sc = copy.deepcopy(ds_all[9])

#####################TOP STUFF HERE####################################################

print "Top Voxels"
topV_t = fast_top_voxels(tc['data'], sc['data'], len(tc['data']))
topV_s = fast_top_voxels(sc['data'], tc['data'], len(sc['data']))

csvout = csv.writer(open('statz.csv', 'w'))

for TV in range(100, 2001, 50):

	num_voxels = TV
	tcc = copy.deepcopy(tc)
	scc = copy.deepcopy(sc)	
	tcc['data'] = tc['data'][topV_t[-num_voxels:]]
	scc['data'] = sc['data'][topV_s[-num_voxels:]]

	#####################PCA STUFF HERE####################################################
	for PC in range(2, 30, 1):
		t = copy.deepcopy(tcc)
		s = copy.deepcopy(scc)	
		print "\n\nPCA\n"
		num_pcs = PC
		pca = PCA(num_pcs)
		pca.fit(tcc['data'].transpose())
		t['data'] = pca.transform(tcc['data'].transpose()).transpose()
		print "Explained Variance T:"
		print numpy.sum(pca.explained_variance_ratio_)

		pca.fit(scc['data'].transpose())
		s['data'] = pca.transform(scc['data'].transpose()).transpose()
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

		T,b = build_xform(s,t)

		s['data'] = sDataTrain.transpose()

		new_s = apply_xform(s,T,b)

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
