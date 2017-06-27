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
	
	print 'Building transform'
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
	
	print 'Building pseudo transform'
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
	
	print 'Building gaussian transform'
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

	print 'Applying transform'
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

############
def load_EVs_data(datapathname):
	import csv

	reader = csv.reader(open(datapathname, 'r'))
	EVs = []

	for r in reader:
		EVs.append( (r[0],r[1], r[2]))

	return EVs

############
def get_labels(TR, EV):
	labels = []
	for i in EV:
		sp = float(i[0])/TR
		ep = sp + float(i[1])/TR	
		ev = i[2]
		for j in range(int(numpy.floor(sp)), int(numpy.ceil(ep))):
			labels.append(ev)

	return labels

############
def get_data_for_class(TR, EV, inData):
	data = []
	for i in EV:
		sp = float(i[0])/TR
		ep = sp + float(i[1])/TR	
		ev = i[2]
		for j in range(int(numpy.floor(sp)), int(numpy.ceil(ep))):
			data.append(inData[j])

	return data

############
def get_labels_no_cue(TR, EV):
	labels = []
	for i in EV:
		if(i[2] == 'cue'):
			continue		
		sp = float(i[0])/TR
		ep = sp + float(i[1])/TR	
		ev = i[2]
		for j in range(int(numpy.floor(sp)), int(numpy.ceil(ep))):
			labels.append(ev)

	return labels

############
def get_data_for_class_no_cue(TR, EV, inData):
	data = []
	for i in EV:
		if(i[2] == 'cue'):
			continue
		sp = float(i[0])/TR
		ep = sp + float(i[1])/TR	
		ev = i[2]
		for j in range(int(numpy.floor(sp)), int(numpy.ceil(ep))):
			data.append(inData[j])

	return data

############
def get_labels_no_cue_cut_edges(TR, EV):
	labels = []
	for i in EV:
		if(i[2] == 'cue'):
			continue		
		sp = float(i[0])/TR
		ep = sp + float(i[1])/TR	
		ev = i[2]
		for j in range(int(numpy.floor(sp) + 4), int(numpy.ceil(ep) - 5)):	#was 5
			labels.append(ev)

	return labels

############
def get_data_for_class_no_cue_cut_edges(TR, EV, inData):
	data = []
	for i in EV:
		if(i[2] == 'cue'):
			continue
		sp = float(i[0])/TR
		ep = sp + float(i[1])/TR	
		ev = i[2]
		for j in range(int(numpy.floor(sp) + 4), int(numpy.ceil(ep) - 5)):	#was5
			data.append(inData[j])

	return data


############
print 'Loading VTX files'
datapath = '/home/james/Desktop/HCP-Processed/'
#datapath = '/media/james/My Passport/HCP/HCP-Processed/'

inputs = xi.python_vts(datapath+'Motor/100307/MNINonLinear/Results/tfMRI_MOTOR_LR/tfMRI_MOTOR_LR.nii',datapath+'primary motor_pFgA_z_FDR_0.01.nii.gz',9)
inputt = xi.python_vts(datapath+'Motor/100408/MNINonLinear/Results/tfMRI_MOTOR_LR/tfMRI_MOTOR_LR.nii',datapath+'primary motor_pFgA_z_FDR_0.01.nii.gz',9)

#s = xi.python_vts(datapath+'WorkingMemory/100408/MNINonLinear/Results/tfMRI_WM_LR/tfMRI_WM_LR.nii.gz',datapath+'primary visual_pFgA_z_FDR_0.01.nii.gz',3)
#t = xi.python_vts(datapath+'WorkingMemory/101006/MNINonLinear/Results/tfMRI_WM_LR/tfMRI_WM_LR.nii.gz',datapath+'primary visual_pFgA_z_FDR_0.01.nii.gz',3)


print 'Top Voxels'

#top_s = fast_top_voxels(inputs['data'],inputt['data'],500)
#top_t = fast_top_voxels(inputt['data'],inputs['data'],500)

#inputs['data'] = inputs['data'][top_s]
#inputt['data'] = inputt['data'][top_t]

print '\n'
print inputs['data'].shape


for i in range(10,100,10):
	print '-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-'	
	for j in range (10,50,5):
		#####################
		print '\n\nPCA----------------------------------------' #was 50

		s = copy.deepcopy(inputs)
		t = copy.deepcopy(inputt)

		num_pcs=i
		print num_pcs	
		pca1 = PCA(num_pcs)
		pca2 = PCA(num_pcs)
		pca1.fit(s['data'].transpose())
		pca2.fit(t['data'].transpose())
		s['data']=pca1.transform(s['data'].transpose()).transpose()
		t['data']=pca2.transform(t['data'].transpose()).transpose()
		#sData = pca1.transform(s['data'].transpose())
		#tData = pca1.transform(t['data'].transpose())
		'''
		num_pcs=50
		pca1 = PCA(num_pcs)
		pca2 = PCA(num_pcs)
		pca1.fit(s['data'])
		pca2.fit(t['data'])
		s['data']=pca1.components_
		t['data']=pca2.components_
		'''


		sData = s['data'].transpose()
		tData = t['data'].transpose()


		#########################
		#CLASSIFICATION STUFF HERE (This is where I would change how I do MVPA stuff --- BETA, VOXELS, ETC.)

		print 'Relevant vol for Classes'

		rawLabels = load_EVs_data(datapath+'MotorEVs.csv')

		labels = get_labels_no_cue_cut_edges(0.720, rawLabels)

		print len(labels)

		sData = get_data_for_class_no_cue_cut_edges(0.720, rawLabels, sData)
		tData = get_data_for_class_no_cue_cut_edges(0.720, rawLabels, tData)

		print len(sData)
		print len(tData)

		myGuy1 = svm.SVC(kernel='linear')
		myGuy2 = KNeighborsClassifier(1)
		
		print
		print j
		D_train, D_test, L_train, L_test = cross_validation.train_test_split(tData, labels, test_size = (float(j)/100), random_state=0)

		myGuy1.fit(D_train, L_train)
		myGuy2.fit(D_train, L_train)

		#print 'SVM-lin target score: ' + str(myGuy1.score(tData, labels))
		#print '1-NN target score: ' + str(myGuy2.score(tData, labels))
		#print 'SVM-lin source score before align: ' + str(myGuy1.score(sData, labels))
		#print '1-NN source score before align: ' + str(myGuy2.score(sData, labels))


		print "\n"
		score = myGuy1.score(D_test, L_test)
		print 'SVM-lin target score test: '
		print score

		score = myGuy1.score(D_test, L_test)
		print '1-NN target score test: '
		print score

		#######
		print "\n"
		score = myGuy1.score(sData, labels)
		#score = cross_validation.cross_val_score(myGuy1, sData, labels, cv=5)
		print 'SVM-lin source score before align: '
		print score

		score = myGuy2.score(sData, labels)
		#score = cross_validation.cross_val_score(myGuy2, sData, labels, cv=5)
		print '1-NN source score before align: '
		print score

		print "\n"


		T,b = build_xform(s,t)
		new_s = apply_xform(s,T,b)

		nsData = new_s['data'].transpose()
		print nsData.shape

		nsData = get_data_for_class_no_cue_cut_edges(0.720, rawLabels, nsData)
		print len(nsData)


		#print 'SVM-lin source score after align: ' + str(myGuy1.score(nsData, labels))
		#print '1-NN source score after align: ' + str(myGuy2.score(nsData, labels))

		print "\n"
		#score = cross_validation.cross_val_score(myGuy1, nsData, labels, cv=5)
		score = myGuy1.score(nsData, labels)
		print 'SVM-lin source score after align: '
		print score

		#score = cross_validation.cross_val_score(myGuy2, nsData, labels, cv=5)
		score = myGuy2.score(nsData, labels)
		print '1-NN source score after align: '
		print score



