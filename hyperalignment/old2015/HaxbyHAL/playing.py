####NEED THIS export PYTHONPATH=$PWD


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


def build_xform(svtx, tvtx):
	"""
	Build a linear transform T,b that maps from the source vtx svtx onto the target vtx
	"""
	
	source = svtx.samples.transpose()
	target = tvtx.samples.transpose()
	
	#print source.shape
	#print target.shape

    #adds a tiiiint amount of random noise so the matrix is not singular
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

	outmat = numpy.zeros(vtx.samples.transpose().shape)
	
	dat = vtx.samples.transpose()
    
	#print dat.shape
	#print T.shape

	for i,func_point in enumerate(dat.transpose()):	
		outmat[:,i] = T.dot(func_point-b[0]) + b[1]
	
	#outvtx = copy.deepcopy(vtx)
	outvtx = vtx	
	outvtx.samples = outmat.transpose()
	#print outmat.shape	

	return outvtx

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


def MoKa_Build(all_data):
	all_T = []
	all_b = []
	t = all_data[0]	
	
	for i in range(1, len(all_data)):
		T,b = build_xform(all_data[i],t)
		all_T.append(T)
		all_b.append(b)

	return all_T, all_b

def MoKa_Apply(all_data, Ts, bs):
	mapped = []
	mapped.append(all_data[0])
	#mapped.append(copy.deepcopy(all_data[0]))	

	for i in range(1, len(all_data)):
		mapped.append(apply_xform(all_data[i],Ts[i-1],bs[i-1]))
	
	return mapped

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

from mvpa2.suite import *
from mvpa2.support.pylab import pl
from mvpa2.misc.data_generators import noisy_2d_fx
from mvpa2.mappers.svd import SVDMapper
from mvpa2.mappers.mdp_adaptor import ICAMapper, PCAMapper
from mvpa2 import cfg
verbose.level = 2

##########################################

verbose(1, "Loading data...")
filepath = os.path.join(pymvpa_datadbroot, 'hyperalignment_tutorial_data_2.4', "hyperalignment_tutorial_data_2.4.hdf5.gz")


ds_all = h5load(filepath)
print ds_all[0].samples.shape
# zscore all datasets individually
_ = [zscore(ds) for ds in ds_all]
# inject the subject ID into all datasets
for i,sd in enumerate(ds_all):
    sd.sa['subject'] = np.repeat(i, len(sd))
# number of subjects
nsubjs = len(ds_all)
# number of categories
ncats = len(ds_all[0].UT)
# number of run
nruns = len(ds_all[0].UC)
verbose(2, "%d subjects" % len(ds_all))
verbose(2, "Per-subject dataset: %i samples with %i features" % ds_all[0].shape)
verbose(2, "Stimulus categories: %s" % ', '.join(ds_all[0].UT))

##########################################

# use same classifier
clf = LinearCSVMC()

# feature selection helpers
nf = 100
fselector = FixedNElementTailSelector(nf, tail='upper',
                                      mode='select',sort=False)
sbfs = SensitivityBasedFeatureSelection(OneWayAnova(), fselector,
                                        enable_ca=['sensitivities'])
# create classifier with automatic feature selection
fsclf = FeatureSelectionClassifier(clf, sbfs)

##########################################

verbose(1, "Performing classification analyses...")
verbose(2, "within-subject...", cr=False, lf=False)
wsc_start_time = time.time()
cv = CrossValidation(fsclf,
                     NFoldPartitioner(attr='chunks'),
                     errorfx=mean_match_accuracy)
# store results in a sequence
wsc_results = [cv(sd) for sd in ds_all]
wsc_results = vstack(wsc_results)
verbose(2, " done in %.1f seconds" % (time.time() - wsc_start_time,))

##########################################

verbose(2, "between-subject (anatomically aligned)...", cr=False, lf=False)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
ds_mni = vstack(ds_all)
mni_start_time = time.time()
cv = CrossValidation(fsclf,
                     NFoldPartitioner(attr='subject'),
                     errorfx=mean_match_accuracy)
bsc_mni_results = cv(ds_mni)
verbose(2, "done in %.1f seconds" % (time.time() - mni_start_time,))

##########################################

verbose(2, "between-subject (hyperaligned)...", cr=False, lf=False)
hyper_start_time = time.time()
bsc_hyper_results = []
# same cross-validation over subjects as before
cv = CrossValidation(clf, NFoldPartitioner(attr='subject'),
                     errorfx=mean_match_accuracy)

#ds_all = ds_all[6:8]
#print
#print len(ds_all)

# leave-one-run-out for hyperalignment training
for test_run in range(nruns):
    # split in training and testing set
    ds_train = [sd[sd.sa.chunks != test_run,:] for sd in ds_all]
    ds_test = [sd[sd.sa.chunks == test_run,:] for sd in ds_all]

    # manual feature selection for every individual dataset in the list
    anova = OneWayAnova()
    fscores = [anova(sd) for sd in ds_train]
    featsels = [StaticFeatureSelection(fselector(fscore)) for fscore in fscores]
    ds_train_fs = [featsels[i].forward(sd) for i, sd in enumerate(ds_train)]


    # Perform hyperalignment on the training data with default parameters.
    # Computing hyperalignment parameters is as simple as calling the
    # hyperalignment object with a list of datasets. All datasets must have the
    # same number of samples and time-locked responses are assumed.
    # Hyperalignment returns a list of mappers corresponding to subjects in the
    # same order as the list of datasets we passed in.


    hyper = Hyperalignment()
    hypmaps = hyper(ds_train_fs)

    # Applying hyperalignment parameters is similar to applying any mapper in
    # PyMVPA. We start by selecting the voxels that we used to derive the
    # hyperalignment parameters. And then apply the hyperalignment parameters
    # by running the test dataset through the forward() function of the mapper.

    ds_test_fs = [featsels[i].forward(sd) for i, sd in enumerate(ds_test)]
    ds_hyper = [ hypmaps[i].forward(sd) for i, sd in enumerate(ds_test_fs)]

    # Now, we have a list of datasets with feature correspondence in a common
    # space derived from the training data. Just as in the between-subject
    # analyses of anatomically aligned data we can stack them all up and run the
    # crossvalidation analysis.

    ds_hyper = vstack(ds_hyper)
    # zscore each subject individually after transformation for optimal
    # performance
    zscore(ds_hyper, chunks_attr='subject')
    res_cv = cv(ds_hyper)
    bsc_hyper_results.append(res_cv)

bsc_hyper_results = hstack(bsc_hyper_results)
verbose(2, "done in %.1f seconds" % (time.time() - hyper_start_time,))

########################################## vDO MoKa HEREv

verbose(2, "between-subject (MoKa-Aligned)...", cr=False, lf=False)
MoKa_start_time = time.time()
bsc_MoKa_results = []

# same cross-validation over subjects as before
cv = CrossValidation(clf, NFoldPartitioner(attr='subject'),
                     errorfx=mean_match_accuracy)

# leave-one-run-out for hyperalignment training
for test_run in range(nruns): 

	# split in training and testing set
    ds_train = [sd[sd.sa.chunks != test_run,:] for sd in ds_all]
    ds_test = [sd[sd.sa.chunks == test_run,:] for sd in ds_all]

    # manual feature selection for every individual dataset in the list
    anova = OneWayAnova()
    fscores = [anova(sd) for sd in ds_train]
    featsels = [StaticFeatureSelection(fselector(fscore)) for fscore in fscores]    
    ds_train_fs = [featsels[i].forward(sd) for i, sd in enumerate(ds_train)]
	
    '''
    num_pcs = 10
    pcas = [PCA(num_pcs) for sd in ds_train]
    PCAmappers = [PCAMapper() for _ in enumerate(ds_train)]######################################3 (zeros and shape?
    ds_train_fs = [PCAmappers[i].forward(sd.samples) for i, sd in enumerate(ds_train)]
    '''

    '''
    num_pcs = 5
    myPCA = PCA(num_pcs)	
    for ds in ds_train_fs:
        #print ds.samples.shape
        myPCA.fit(ds.samples)
        ds.samples = myPCA.transform(ds.samples)	
    '''
    #for ds in ds_train_fs:
        #print ds.samples.shape
        #ds.fa = FeatureAttributesCollection(length=num_pcs)
        #ds.fa.update(ds.fa)
        #ds.fa.length = num_pcs
        #ds.fa.voxel_indices = numpy.zeros([num_pcs, 3])

    Ts,bs = MoKa_Build(ds_train_fs)
	

    #print ds_test[0].samples.shape

    fscores = [anova(sd) for sd in ds_test]
    featsels = [StaticFeatureSelection(fselector(fscore)) for fscore in fscores]
    ds_test_fs = [featsels[i].forward(sd) for i, sd in enumerate(ds_test)]
    #print ds_test_fs[0].samples.shape

    '''
    PCAmappers = [PCAMapper(10) for _ in enumerate(ds_train)]
    ds_train_fs = [PCAmappers[i].forward(sd.samples) for i, sd in enumerate(ds_train)]
    '''

    '''
    for ds in ds_test_fs:
        myPCA.fit(ds.samples)
        ds.samples = myPCA.transform(ds.samples)
    '''
    #print ds_test_fs[0].samples.shape

    #for ds in ds_train_fs:
        #ds.fa = FeatureAttributesCollection(length=num_pcs)
        #ds.fa.update(ds.fa)
        #ds.fa.length = num_pcs
        #ds.fa.voxel_indices = numpy.zeros([num_pcs, 3])

    #print ds_test_fs[0].samples.shape    

    MoKa_mapped = MoKa_Apply(ds_test_fs, Ts, bs)

    MoKa_mapped = vstack(MoKa_mapped)
    zscore(MoKa_mapped, chunks_attr='subject')
    res_cv = cv(MoKa_mapped)
    bsc_MoKa_results.append(res_cv)

bsc_MoKa_results = hstack(bsc_MoKa_results)
verbose(2, "done in %.1f seconds" % (time.time() - MoKa_start_time,))


########################################## ^DO MoKa HERE^

verbose(1, "Average classification accuracies:")
verbose(2, "within-subject: %.2f +/-%.3f"
        % (np.mean(wsc_results),
           np.std(wsc_results) / np.sqrt(nsubjs - 1)))
verbose(2, "between-subject (anatomically aligned): %.2f +/-%.3f"
        % (np.mean(bsc_mni_results),
           np.std(np.mean(bsc_mni_results, axis=1)) / np.sqrt(nsubjs - 1)))
verbose(2, "between-subject (hyperaligned): %.2f +/-%.3f" \
        % (np.mean(bsc_hyper_results),
           np.std(np.mean(bsc_hyper_results, axis=1)) / np.sqrt(nsubjs - 1)))
verbose(2, "between-subject (MoKa-Aligned): %.2f +/-%.3f" \
        % (np.mean(bsc_MoKa_results),
           np.std(np.mean(bsc_MoKa_results, axis=1)) / np.sqrt(nsubjs - 1)))
##########################################


