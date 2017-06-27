# Using the Monge-Kantorovitch solution to hyperalign fMRI data
####NEED THIS export PYTHONPATH=$PWD


"""
THIS ONE DOES IT FOR SHUFFLED OF 7!
THIS ONE ALSO CHANGES THE ONE BEING TESTED IN THE MAPPING
THIS ONE ALSO CHANGES WHO IS BEING MAPPED TO

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

for tTo in range(0, len(ds_all)):

	swap_ds = ds_all[tTo]		
	ds_all[tTo] = ds_all[0]
	ds_all[0] = swap_ds

	#ds_all = ds_all[0:3]

	tc = [copy.deepcopy(ds_all[0]) for ds in ds_all[1:]]
	sc = [copy.deepcopy(ds) for ds in ds_all[1:]]

	#####################TOP STUFF HERE####################################################

	print "Top Voxels"
	topV_t = []
	topV_s = []
	for i in range(0, len(sc)):
		topV_t.append(fast_top_voxels(tc[i]['data'], sc[i]['data'], len(tc[i]['data'])))
		topV_s.append(fast_top_voxels(sc[i]['data'], tc[i]['data'], len(sc[i]['data'])))

	csvout = csv.writer(open('statz5-' + str(tTo) + '.csv', 'w'))

	#num_voxels = 100
	#3rd and 4th to 500 and 150
	for TV in range(40, 2001, 1):

		num_voxels = TV
		tcc = [copy.deepcopy(X) for X in tc]
		scc = [copy.deepcopy(X)	for X in sc]
		for i in range(0, len(sc)):
			tcc[i]['data'] = tc[i]['data'][topV_t[i][-num_voxels:]]
			scc[i]['data'] = sc[i]['data'][topV_s[i][-num_voxels:]]

		#####################PCA STUFF HERE####################################################

		for PC in range(2, 35, 1):
			t = [copy.deepcopy(X) for X in tcc]
			s = [copy.deepcopy(X) for X in scc]
			print "\n\nPCA"
			num_pcs = PC
			pca = PCA(num_pcs)
			tData = []
			sData = []

			print str(TV) + "\t" + str(PC)
			for i in range(0, len(s)):
				pca.fit(t[i]['data'].transpose())
				t[i]['data'] = pca.transform(t[i]['data'].transpose()).transpose()

				pca.fit(s[i]['data'].transpose())
				s[i]['data'] = pca.transform(s[i]['data'].transpose()).transpose()

				tData.append(t[i]['data'].transpose())
				sData.append(s[i]['data'].transpose())


			#####################PART STUFF HERE###################################################	

			score1 = []
			score2 = []

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

					T,b = build_xform(s[i],t[i])

					s[i]['data'] = sDataTrain[i].transpose()

					new_s.append(apply_xform(s[i],T,b))

					new_sData.append(new_s[i]['data'].transpose())

				#####################PREP STUFF HERE###################################################

				for sFrom in range(0, len(s)):	#for changing who is being tested in the map
					SA = sDataTrain[sFrom]
					FA = new_sData[sFrom]

					TRAIN = []
					TRAIN.append(tDataTrain[sFrom])
					LABELS = []
					LABELS.append(tlabels)
					for i in range(0, len(s)):
						if(sFrom != i):					
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


			print "SVM-FA"
			print numpy.mean(score1)
			print "KNN-FA"
			print numpy.mean(score2)
			print "\n"

			csvout.writerow([TV, PC, numpy.mean(score1), numpy.mean(score2)])
