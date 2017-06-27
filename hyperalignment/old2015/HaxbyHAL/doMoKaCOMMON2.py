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

# Build affine transformation matrices with pseudo inverse
#@jit(argtypes=[double[:], double[:,:], double[:,:]])
def build_xform_pseudo(svtx, tvtx):
	"""
	Build a linear transform T,b that maps from the source vtx svtx onto the target vtx with a pseudo inverse matrix
	"""
	
	#print 'Building pseudo transform'
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
	T = numpy.linalg.pinv(numpy.real(scipy.linalg.sqrtm(s_s))).dot( numpy.real(scipy.linalg.sqrtm(( numpy.real(scipy.linalg.sqrtm(s_s))).dot(s_t).dot(numpy.real(scipy.linalg.sqrtm(s_s)))))).dot(numpy.linalg.pinv(numpy.real(scipy.linalg.sqrtm(s_s))))

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

def fast_top_voxels_SCORE(xmat,ymat,topn=50):	
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
	#indx = numpy.argsort(score)
	
	return score

def fast_top_voxels_SCORE_META(xmat,ymats,topn=50):
	scores = [fast_top_voxels_SCORE(xmat,ymat,topn) for ymat in ymats] 
	scores = numpy.array(scores)	
	#print scores	
	scores = numpy.mean(scores, axis=0)
	indx = numpy.argsort(scores)
	return indx[-topn:]

############
print 'Loading files'

from mvpa2.suite import *
from mvpa2.support.pylab import pl
from mvpa2.misc.data_generators import noisy_2d_fx
from mvpa2.mappers.svd import SVDMapper
from mvpa2.mappers.mdp_adaptor import ICAMapper, PCAMapper
from mvpa2 import cfg


##########################################

print "Loading Data"
filepath = os.path.join(pymvpa_datadbroot, 'hyperalignment_tutorial_data_2.4', "hyperalignment_tutorial_data_2.4.hdf5.gz")
#filepath = "/home/james/Dropbox/SchoolStuff/UWO/FunctionalAlignment/hyperalignment/HaxbyHAL/datadb/hyperalignment_tutorial_data_2.4/hyperalignment_tutorial_data_2.4.hdf5.gz"

ds_all_IN = h5load(filepath)

labels =  ds_all_IN[0].targets

#haxx
ds_all = []
for ds in ds_all_IN:
	ds_all.append(xi.pack_vtx(numpy.array(ds.samples.transpose()),'null','VTS','Voxel Time Series [channels,volumes]','null','null','null'))

print ds_all[0]['data']

print 'Top Voxels'

top_voxels_S = []


for i in range(1, len(ds_all)):
	top_voxels_S.append(fast_top_voxels(ds_all[i]['data'], ds_all[0]['data'], len(ds_all[0]['data'])))
	#top_voxels_T.append(fast_top_voxels(ds_all[0]['data'], ds_all[i]['data'], len(ds_all[0]['data'])))

todoles = []
for ds in ds_all[1:]:
	todoles.append(ds['data'])

top_voxels_t = fast_top_voxels_SCORE_META(ds_all[0]['data'], todoles, len(ds_all[0]['data']))						


#top_voxels_S = [fast_top_voxels(ds_all[0]['data'], ds_all[i]['data'], len(ds_all[0]['data'])) for i in range(1, len(ds_all))]
#top_voxels_T = [fast_top_voxels(ds_all[i]['data'],ds_all[0]['data'],len(ds_all[0]['data'])) for i in range(1, len(ds_all))]

#top_voxels_S = numpy.array(top_voxels_S)
#top_voxels_T = numpy.array(top_voxels_T)

#top_voxels_S = [fast_top_voxels(ds_all[0]['data'], ds_all[i]['data'], len(ds_all[0]['data'])) for i in range(1, 3)]
#top_voxels_T = [fast_top_voxels(ds_all[i]['data'],ds_all[0]['data'],len(ds_all[0]['data'])) for i in range(1, 3)]

#top_s = fast_top_voxels(ds_all[3]['data'],ds_all[0]['data'], len(ds_all[0]['data']))
#top_t = fast_top_voxels(ds_all[0]['data'],ds_all[3]['data'], len(ds_all[0]['data'])) 

maxSTT = [0] * len(top_voxels_S)
maxKTT = [0] * len(top_voxels_S)
maxST = [0] * len(top_voxels_S)
maxKT = [0] * len(top_voxels_S)
maxSS = [0] * len(top_voxels_S)
maxKS = [0] * len(top_voxels_S)
maxSF = [0] * len(top_voxels_S)
maxKF = [0] * len(top_voxels_S)

csvout = csv.writer(open('statz.csv', 'w'))
for topz in range(100, 1500, 100):
#for topz in range(400, 500, 100):

	csvout.writerow([' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',topz])
	#Ss = [copy.deepcopy(ds) for ds in ds_all]
	#Ts = [copy.deepcopy(ds) for ds in ds_all]
	
	#Ss = numpy.array(Ss)
	#Ts = numpy.array(Ts)

	Ssb = []			#ahh read "ess-is"
	#Ts = []

	for k in range(1, len(ds_all)):
		Ssb.append(copy.deepcopy(ds_all[k]))
		#Ts.append(copy.deepcopy(ds_all[k]))



	for k in range(0, len(Ssb)):
		Ssb[k]['data'] = Ssb[k]['data'][top_voxels_S[k][-topz:]]
		#Ts[k]['data'] = Ts[k]['data'][top_voxels_T[k][-topz:]]

	tb = copy.deepcopy(ds_all[0])
	tb['data'] = tb['data'][top_voxels_t[-topz:]]	

	
	
	for i in range(8,35,1):
	#for i in range(15,16,1):
		bleh = []
		print '-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-'	
		for j in range (20,60,10):
		#for j in range (30,40,10):
			print topz
			#####################
			print '\n\nPCA----------------------------------------' #was 50

			t = copy.deepcopy(tb)
			Ss = copy.deepcopy(Ssb)

			#s['data'] = s['data'][top_voxels_S[0][-topz:]]
			#t['data'] = t['data'][top_voxels_T[0][-topz:]]

			#t['data'] = t['data'][top_t[-topz:]]


			num_pcs=i
			print num_pcs	
			pca = PCA(num_pcs)

			pca.fit(t['data'].transpose())
			t['data'] = pca.transform(t['data'].transpose()).transpose()

			for k in range(0, len(Ss)):
				
				pca.fit(Ss[k]['data'].transpose())
				Ss[k]['data'] = pca.transform(Ss[k]['data'].transpose()).transpose()

			#sData = []
			sData = []
	
			tData = t['data'].transpose()
			for k in range(0, len(Ss)):
				sData.append(Ss[k]['data'].transpose())							
				#tData.append(Ts[k]['data'].transpose())
			#sData = [s['data'].transpose() for s in Ss]
			#tData = [t['data'].transpose() for t in Ts]
			
			sData = numpy.array(sData)
			#tData = numpy.array(tData)

			#########################

			myGuy1 = svm.SVC(kernel='linear')
			myGuy2 = KNeighborsClassifier(1)
			
			sTST = []
			sTKT = []
 			sTS = []
			sTK = []
			sSS = []
			sSK = []
			sFS = []
			sFK = []
		

			print
			print j

			for e in range(0, len(sData)):

				tLongData = []
				labelsLong = []
				tLongData.append(tData)
				labelsLong.append(labels)
				for k in range(0, len(sData)):
					if k !=e:
						T,b = build_xform(Ss[k],t)
						new_map = apply_xform(Ss[k],T,b)
						nMapData = new_map['data'].transpose()
						tLongData.append(nMapData)
						labelsLong.append(labels)



				tLongData = numpy.vstack(tLongData)
				labelsLong = numpy.concatenate(labelsLong)

				#print tLongData.shape
				#print
				#print labelsLong.shape
				

				D_train, D_test, L_train, L_test = cross_validation.train_test_split(tLongData, labelsLong, test_size = (float(j)/100), random_state=0)

				myGuy1.fit(D_train, L_train)
				myGuy2.fit(D_train, L_train)
				
				
				sTST.append(myGuy1.score(D_train, L_train))			
				
				sTKT.append(myGuy2.score(D_train, L_train))

				sTS.append(myGuy1.score(D_test, L_test))

				sTK.append(myGuy2.score(D_test, L_test))

				sSS.append(myGuy1.score(sData[e], labels))

				sSK.append(myGuy2.score(sData[e], labels))


				T,b = build_xform(Ss[e],t)
				new_s = apply_xform(Ss[e],T,b)

				nsData = new_s['data'].transpose()

				score = myGuy1.score(nsData, labels)
				sFS.append(score)
				if score > maxSF[e]:
					maxSTT[e] = myGuy1.score(D_train, L_train)					
					maxST[e] = myGuy1.score(D_test, L_test)					
					maxSS[e] = myGuy1.score(sData[e], labels)					
					maxSF[e] = score
				
				score = myGuy2.score(nsData, labels)
				sFK.append(score)
				if score > maxKF[e]:
					maxKTT[e] = myGuy2.score(D_train, L_train)					
					maxKT[e] = myGuy2.score(D_test, L_test)					
					maxKS[e] = myGuy2.score(sData[e], labels)						
					maxKF[e] = score

			print "\n"
			score = numpy.mean(sTST)
			print score
			print 'SVM-lin target score train: '
			score = numpy.mean(sTKT)
			print '1-NN target score train: '
			print score

			print "\n"
			score = numpy.mean(sTS)
			print score
			print 'SVM-lin target score test: '
			score = numpy.mean(sTK)
			print '1-NN target score test: '
			print score

			#######
			print "\n"
			score = numpy.mean(sSS)

			print 'SVM-lin source score before align: '
			print score

			score = numpy.mean(sSK)

			print '1-NN source score before align: '
			print score

			print "\n"

			print "\n"
			#score = cross_validation.cross_val_score(myGuy1, nsData, labels, cv=5)
			score = numpy.mean(sFS)
			print 'SVM-lin source score after align: '
			print score
			bleh.append(score)
			#score = cross_validation.cross_val_score(myGuy2, nsData, labels, cv=5)
			score = numpy.mean(sFK)
			print '1-NN source score after align: '
			print score

			bleh.append(score)

		
			'''
			##################################
			pred_SVM = myGuy1.predict(D_test);
			pred_KNN = myGuy2.predict(D_test);

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

			pred_SVM = myGuy1.predict(sData);
			pred_KNN = myGuy2.predict(sData);

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


			pred_SVM = myGuy1.predict(nsData);
			pred_KNN = myGuy2.predict(nsData);

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
		csvout.writerow(bleh)

print "MAX FOR SVMS TRAIN"
print maxSTT
print numpy.mean(maxSTT)


print "MAX FOR KNN TRAIN"
print maxKTT
print numpy.mean(maxKTT)


print "MAX FOR SVMS TEST"
print maxST
print numpy.mean(maxST)


print "MAX FOR KNN TEST"
print maxKT
print numpy.mean(maxKT)


print "MAX FOR SVMS B4"
print maxSS
print numpy.mean(maxSS)


print "MAX FOR KNN B4"
print maxKS
print numpy.mean(maxKS)


print "MAX FOR SVMS FA"
print maxSF
print numpy.mean(maxSF)


print "MAX FOR KNN FA"
print maxKF
print numpy.mean(maxKF)

