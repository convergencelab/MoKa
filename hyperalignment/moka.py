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

import copy
import numpy
import scipy
import scipy.weave

def build_xform(svtx, tvtx):
	"""
	Build a linear transform T,b that maps from the source vtx svtx onto the target vtx
	"""
	
	
	try:
		source = svtx['data']
		target = tvtx['data']
	except:
		source = svtx
		target = tvtx

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
	try:
		source = svtx['data']
		target = tvtx['data']
	except:
		source = svtx
		target = tvtx

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
	Build a linear transform T,b that maps from the source vtx svtx onto the target vtx with tinnnny gaussian noise added to the matrix to eliminate singularity issues.
	"""
	
	#print 'Building gaussian transform'
	try:
		source = svtx['data']
		target = tvtx['data']
	except:
		source = svtx
		target = tvtx
	
		
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


	#commented out stuff is for when working with actual xifti things

	try:
		vtx = vtx['data']
	except:
		vtx = vtx


	#print 'Applying transform'
	#outmat = numpy.zeros(vtx['data'].shape)
	outmat = numpy.zeros(vtx.shape)

	#dat = vtx['data']
	dat = vtx	

	for i,func_point in enumerate(dat.transpose()):	
		outmat[:,i] = T.dot(func_point-b[0]) + b[1]
	
	#outvtx = copy.deepcopy(vtx)
	#outvtx['data'] = outmat
	
	#return outvtx
	return outmat

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

# Find the 'top voxels' as per Haxby paper
# DEPRECATED -- left here for algorithmic clarity to non-C programmers.
# This is unbearably slow for production use... use the weaved version below.
#
# COULD BE BETTER IF WE USED numpy.corrcoef ???
#
def top_voxels(s,t,topn=50):
	score = numpy.zeros(s.shape[0])
	
	for i,voxel in enumerate(s):
		maxscore = 0
		print i
		for target in t:
			c=scipy.stats.pearsonr(voxel,target)[0]
			if c > maxscore:
				maxscore = c
		score[i] = maxscore
	
	indx = numpy.argsort(score)
	return indx[-topn:]

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
