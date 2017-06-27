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
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, FastICA


from numba import autojit,jit,double,uint32

#datapath='/Users/daley/dataForOthers/Casey/'

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

# FIXME return a vtx, not a matrix
def apply_xform(vtx,T,b):
	"""
	Given Monge-Kantrovich linear transformation defined by T,b; map the given vtx into
	the space defined by the transform.
	"""

	print 'Applying transform'
	outmat = numpy.zeros(vtx['data'].shape)
	
	dat = vtx['data']
	
	for i,func_point in enumerate(dat.transpose()):		#WHY .transpose? b.c. how structure is defined?
		outmat[:,i] = T.dot(func_point-b[0]) + b[1]
	
	outvtx = copy.deepcopy(vtx)
	outvtx['data'] = outmat
	
	return outvtx


# Find the 'top voxels' as per Haxby paper
# DEPRECATED -- left here for algorithmic clarity to non-C programmers.
# This is unbearably slow for production use... use the weaved version below.
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

# USE THIS ONE. 60% OF THE TIME IT WORKS EVERY TIME.	
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
	
	
	
	
#s = xi.python_vts(datapath+'S1.nii.gz',datapath+'sphere2.nii.gz')
#t = xi.python_vts(datapath+'S2.nii.gz',datapath+'sphere2.nii.gz')

print 'Loading VTX files'
maskfile = './S2/segment/gm2func.nii.gz'
datapath = './'

s = xi.python_vts(datapath+'S1/func/func_res_4mm_std.nii.gz','gm4mm.nii.gz',150)
t = xi.python_vts(datapath+'S2/func/func_res_4mm_std.nii.gz','gm4mm.nii.gz',150)

print s['data'].shape
print t['data'].shape

# FIXME HACK for quick test
s['data'] = s['data'][0:200,:]
t['data'] = t['data'][0:200,:]

# Dimensionality reduction by PCA _or_ "top voxels"?
doPCA = False
if doPCA:
	num_pcs=30
	pca1 = PCA(num_pcs)
	pca2 = PCA(num_pcs)
	pca1.fit(s['data'])
	pca2.fit(t['data'])
	s['data']=pca1.components_
	t['data']=pca2.components_
else:
	# Take top 500 voxels
	top_s = fast_top_voxels(s['data'],t['data'],50)		#500? this looks like 50 to me...
	top_t = fast_top_voxels(t['data'],s['data'],50)
	s['data'] = s['data'][top_s]
	t['data'] = t['data'][top_t]

	#num_pcs=40
	#pca1 = PCA(num_pcs)
	#pca2 = PCA(num_pcs)
	#pca1.fit(s['data'].transpose())
	#pca2.fit(t['data'].transpose())
	#s['data']=pca1.transform(s['data'].transpose()).transpose()
	#t['data']=pca2.transform(t['data'].transpose()).transpose()


T,b = build_xform_gauss(s,t)

new_s = apply_xform(s,T,b)

#print 'Output? :/'
#xi.mni_viz(new_s,range(200),'gm4mm.nii.gz')

# Applying the xform: (T.dot(input_vector-b[0]) + b[1])

