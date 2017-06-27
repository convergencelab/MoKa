# Xifti 0.1 -- first step evolution from .vtx

# If you're using the standard Poldrack-derived pipeline... things to note:
#
# If you want to work in subject space, you need:
# func/func_res.nii.gz   &&   segment/gm2func.nii.gz
#
# If you want to work in MNI152 space, you need:
# func/func_res2standard.nii.gz  &&  segment/gm2standard.nii.gz

# e.g.
# basename='/Users/daley/newJJ/S2/'
# v = vtx.python_vts(basename+'func/func_res2standard.nii.gz', basename+'segment/gm2standard.nii.gz', maskthresh=0.5)

# TODO for xifti.py:
# - Finish design. What do we NEED?
# - Add CIFTI support
# Uhh... maybe the first one is a subset of the second one if we can figure out
# CIFTI 'parcels'

import nibabel
import itertools
import numpy
import copy

def mni_viz(vtx,slices,ref_file='/Users/daley/Dropbox/xifti/mniref.nii.gz'):
	"""
	Visualize a slice of a VTS-style Xifti file using the internally-stored MNI coordinate list.
	
	:param vtx: Xifti structure to visualize
	:param slices: a list 'Slice'/Volume numbers to blat out
	:param ref_file: Reference file; the output will be blatted into a space of identical dimensions.
	                 Note that this reference file MUST have an sform that maps into MNI152!
	
	:returns: A nibabel image object containing the visualization
	"""
	
	# Load the reference image that we want to blast on to
	ref = nibabel.load(ref_file)
	refshape = (ref.get_data()).shape[0:3]
	
	newdata = numpy.zeros((refshape[0], refshape[1], refshape[2], len(slices)))
	
	# FIXME DEBUG -- blast best community structure only
	xform = numpy.linalg.inv(ref.get_affine())
	
	for slice in slices:
		for idx,mniloc in enumerate(vtx['mnilist']):
			loc = numpy.dot(xform,mniloc)
	
			try:
				newdata[int(loc[0]),int(loc[1]),int(loc[2]),slice] = vtx['data'][:,slice][idx]

			except:
				#print idx, [loc[0],loc[1],loc[2],slice]
				pass
				
	# Stuff new image into a Nifti file and save
	img = nibabel.Nifti1Image(newdata,ref.get_affine(),header=ref.get_header())
	img.to_filename('test.nii.gz') # FIXME DEBUG
	return img

def poisson_ROI(funcfile, maskfile, maskthresh=0.5,num_rois=100, min_dist=5):
	"""
	Extracts ROI-based average timecourses from fMRI data and puts it into a Xifti
	object. ROIs are chosen by Poisson Disk sampling, rather than prespecification.
	"""
	
	if funcfile[-3:] == '.gz':
		print '\n\n *** WARNGING ***: nibabel chokes (silently) on large, compressed, nifti files'
		print 'If', funcfile
		print 'is largeish, you should UNCOMPRESS before trying to load it this way.'
		
	
	raw = nibabel.load(funcfile)
	rawdata = raw.get_data()
	
	mask = nibabel.load(maskfile)
	maskdata = mask.get_data()	
	xform = mask.get_affine()
	
	head=mask.get_header()
	
	try:
		assert head['sform_code'] == 4
	except:
		print '\n\nDANGER: Your maskfile affine transformation (apparently) does not go to MNI 152 space. Visualizations will probably suck.'

	channel_list = []
	mni_list = []
	coord_list =[]
		
	#print 'Incoming shape ', maskdata.shape[2],maskdata.shape[1],maskdata.shape[0]
	for z,y,x in itertools.product(range(maskdata.shape[2]),range(maskdata.shape[1]),range(maskdata.shape[0])):
		if maskdata[x,y,z] > maskthresh:
			if len(rawdata.shape) > 3:
				channel_list.append(rawdata[x,y,z,:])
			else:
				channel_list.append(rawdata[x,y,z])#.reshape(1,1))
			mni_list.append(numpy.dot(xform,numpy.array([x,y,z,1])))
			coord_list.append([x,y,z])
	
	
	# Pick some random seed points
	seed_idx = [ numpy.random.randint(0,len(coord_list)) for i in range(num_rois)]
	
	# Too close?, remove it!
	for i,s in enumerate(seed_idx):
		for j, t in enumerate(seed_idx[i:]):
			if numpy.linalg.norm(numpy.array(coord_list[s])-numpy.array(coord_list[t])) < min_dist:
				seed_idx.remove(t)
	
	# FIXME while loop to keep trying until we get the right number of ROIs.
	while len(seed_idx) < num_rois:
		s = numpy.random.randint(0,len(coord_list))
		seed_idx.append(s)
		print s
		
		for j, t in enumerate(seed_idx[:-1]):
			if numpy.linalg.norm(numpy.array(coord_list[s])-numpy.array(coord_list[t])) < min_dist:
				seed_idx.pop()
				break
		print len(seed_idx)
	
	# FIXME -- should be taking averages over regions, not just single voxels!
	
	channel_list=numpy.array(channel_list)
	mni_list = numpy.array(mni_list)
	
	channel_list = channel_list [ seed_idx ]
	mni_list = mni_list[ seed_idx ]
	
	print 'FIXME NOT FINISHED!!!'
		
	# vts[num_channels, num_TRs]
	vts = channel_list
	
	# VTX is what I'm calling the new structure which includes the VTS _and_ supplementary information.
	# Yes, I should've done this in the first place. I'm an idiot.
	
	return pack_vtx(vts,None,'VTS','Poisson Sampled Voxel Time Series [channels,volumes]',funcfile,maskfile,maskthresh,mnilist=numpy.array(mni_list))
	

	
	   
def python_ROI(funcfile, roifile='/Users/daley/Dropbox/xifti/tcorr05_2level_all.nii.gz', roilevel=15, mni_roi=False):
	"""Extracts ROI-based average timecourses from fMRI data and puts it into a Xifti
	   object. The ROI map itself will be placed in the 'mask' variable and the MNI
	   list will contain the ROI centroids.
	
	:param funcfile: functional neuromaging file in MNI 152 4mm space
	:param roifile: ROI file in MNI 152 4mm space
	:param roilevel: which volume of the ROI file to use? (We assume that
	       each volume represents a different ROI parcellation)
	:param mni_roi: If True, use MNI mappings into roifile instead of direct voxel-to-voxel
	
	:returns: a Xifti file
	"""

	func=nibabel.load(funcfile)
	rois=nibabel.load(roifile)
	
	# Make sure both files have the same volumetric shape and that the
	# ROI file is discrete
	if not mni_roi:
		assert rois.shape[0:3] == func.shape[0:3]
		
	assert rois.get_data_dtype() == 'int16'
	
	# FIXME is there a clear speed benefit to doing this? Check.
	funcdata = func.get_data()
	roidata = rois.get_data()
	
	func_xform = func.get_affine()
	xform = rois.get_affine()
	
	# ROI timeseries
	roi_TS = numpy.zeros([numpy.max(roidata[:,:,:,roilevel]+1),funcdata.shape[3]])
	
	# ROI average counts (yes I'm using parallel arrays; guess why?)
	roi_count = numpy.zeros(roi_TS.shape[0])
	
	mni_list = [numpy.array([0,0,0,0])]*roi_TS.shape[0]
	
	# This split could've been done more elegantly... but it's a posthoc change and I need
	# it done in 10 minutes... so... hold yer nose
	
	if not mni_roi:
		# Iterate over both matrices in parallel
		for x,y,z in itertools.product(range(roidata.shape[0]),range(roidata.shape[1]),range(roidata.shape[2])):
			index = roidata[x,y,z,roilevel]
			if index != 0:
				roi_TS[index] += funcdata[x,y,z,:]
				mni_list[index] = mni_list[index] + numpy.dot(xform,numpy.array([x,y,z,1]))
				roi_count[index] += 1
	else:
		for x,y,z in itertools.product(range(funcdata.shape[0]),range(funcdata.shape[1]),range(funcdata.shape[2])):
			
			func_to_mni = numpy.dot(func_xform,numpy.array([x,y,z,1]))
			mni_to_roi = numpy.dot(numpy.linalg.inv(xform),func_to_mni)
			index = roidata[int(mni_to_roi[0]),int(mni_to_roi[1]),int(mni_to_roi[2]),roilevel]
			
			if index != 0:
				roi_TS[index] += funcdata[x,y,z,:]
				mni_list[index] = mni_list[index] + func_to_mni
				roi_count[index] += 1
						
	# Average each ROI timeseries
	for i in range(1,roi_count.shape[0]):
		if roi_count[i] > 0:
			roi_TS[i] /= roi_count[i]
			mni_list[i] /= roi_count[i]
	
	# FIXME -- should return proper ROI mask
	return pack_vtx(roi_TS[1:],None,'VTS/ROI','ROI-based Voxel Time Series [channels,volumes]',funcfile,roifile,0,mnilist=numpy.array(mni_list))
	




def sub_VTT_from_mask(vtx,submaskfile,shape='square'):
	"""Pull out a sub matrix from a square connection matrix modulo a mask
	   file. The resultant submatrix will be from the intersection of the local Xifti mask and
	   the submask.
	   
	   :param vtx: A connection matrix-type Xifti object
	   :param submaskfile: filename for the desired submask
	   :param shape: Extract a 'square' connection matrix or an 'nxm' raw data matrix
	   
	   :returns: a new Xifti with the submatrix as the data payload and a new mask which is the intersection of the original masks
	"""
	import copy
	
	assert vtx['type'][0:3]=='VTT'
	
	mask = vtx['mask']
	maskdata = mask.get_data()	
	
	submask = nibabel.load(submaskfile)
	submaskdata = submask.get_data()	
	
	newmaskdata = numpy.zeros( submaskdata.shape )
	
	ptr=0
	
	# Make a list of indices to extract from the full VTT instead of
	# doing the extraction in a Python loop with copying (e.g. numpy.vstack)
	subvtt_list = []
	
	for z,y,x in itertools.product(range(maskdata.shape[2]),range(maskdata.shape[1]),range(maskdata.shape[0])):
		if maskdata[x,y,z] > vtx['maskthresh']:
			if submaskdata[x,y,z] > vtx['maskthresh']: #FIXME
				subvtt_list.append(ptr)
				newmaskdata[x,y,z] = 1
			ptr+=1

	# extract all at once! (phast)
	if shape=='square':
		subvtt = vtx['data'][subvtt_list,:][:,subvtt_list]
		newtype = vtx['type']
		newdesc=vtx['desc']
	else:
		subvtt = vtx['data'][subvtt_list,:]
		newtype = 'VTS'
		newdesc = 'Rows of interest masked out from a full VTT [mask channels, full VTT channels]'

	# Create intersected mask image
	newmask = nibabel.nifti1.Nifti1Image(newmaskdata,mask.get_affine(),mask.get_header())
	
	subvtx = pack_vtx(subvtt,newmask,newtype,newdesc,vtx['funcfile'],vtx['maskfile']+' /\ '+submaskfile,vtx['maskthresh'],vtx['mode'])
		
	return subvtx
	
	
# TODO:
# get_subVTT_volume

	
def python_vts(funcfile, maskfile, maskthresh=0.5):
	"""
	Given a functional file and a grey matter mask for that file, strip out the masked
	timeseries into a raw data format. That is, a matrix where the rows are 'channels'
	(voxels) and the columns are 'samples' (TRs / Volumes)
	
	:param funcfile: A (preprocessed!) functional scan
	:param maskfile: A binary grey matter mask with the same dimensions as funcfile
	:param maskthresh: Grey matter intensity threshold for mask inclusion (default 0.5)
	
	:returns: a Xifti structure
	"""
	
	if funcfile[-3:] == '.gz':
		print '\n\n *** WARNGING ***: nibabel chokes (silently) on large, compressed, nifti files'
		print 'If', funcfile
		print 'is largeish, you should UNCOMPRESS before trying to load it this way.'
		
	
	raw = nibabel.load(funcfile)
	rawdata = raw.get_data()
	
	mask = nibabel.load(maskfile)
	maskdata = mask.get_data()	
	xform = mask.get_affine()
	
	head=mask.get_header()
	
	try:
		assert head['sform_code'] == 4
	except:
		print '\n\nDANGER: Your maskfile affine transformation (apparently) does not go to MNI 152 space. Visualizations will probably suck.'
	
	# Make a 'vts'
	channel_list = []
	mni_list = []
	
	#print 'Incoming shape ', maskdata.shape[2],maskdata.shape[1],maskdata.shape[0]
	for z,y,x in itertools.product(range(maskdata.shape[2]),range(maskdata.shape[1]),range(maskdata.shape[0])):
		if maskdata[x,y,z] > maskthresh:
			if len(rawdata.shape) > 3:
				channel_list.append(rawdata[x,y,z,:])
			else:
				channel_list.append(rawdata[x,y,z])#.reshape(1,1))
			mni_list.append(numpy.dot(xform,numpy.array([x,y,z,1])))
	
	# vts[num_channels, num_TRs]
	vts = numpy.array(channel_list)
	
	# VTX is what I'm calling the new structure which includes the VTS _and_ supplementary information.
	# Yes, I should've done this in the first place. I'm an idiot.
	
	return pack_vtx(vts,mask,'VTS','Voxel Time Series [channels,volumes]',funcfile,maskfile,maskthresh,mnilist=numpy.array(mni_list))
	

# FIXME TEMP
def correlation(vtx_in,startTR=0,endTR=-1):
	"""
	Compute the correlation matrix for a given set of timeseries between two TRs and store it in 
	the Xifti file.
	#FIXME overwrite or add?
	"""
	
	if endTR == -1:
		endTR = vtx_in['data'].shape[1]
	print startTR,endTR
	vtt=numpy.corrcoef(vtx_in['data'][:,startTR:endTR])
	
	vtt = numpy.nan_to_num(vtt)
	
	return pack_vtx(vtt, vtx_in['mask'], 'VTT/corrcoef', 'Correlation matrix [channels,channels]',vtx_in['funcfile'],vtx_in['maskfile'],vtx_in['maskthresh'],mnilist=vtx_in['mnilist'])





## I know this stuff doesn't belong here... get rid of it to it's own file posthaste

def row_entropy(row):
	""" Compute the entropy of a single row from a dynamic entropy matrix """
	
	return -numpy.sum(row*log(row))
	
# FIXME Temporary -- fix thresholding, etc.
def dynamic_entropy(vtx_in,windowSize=50,stepSize=50,threshold=0.5):
	"""Compute the per-voxel dynamic network entropy"""
	
	# Array to hold the sum of the total number of times a pair of edges was connected,
	# over all windows
	counts = numpy.zeros((vtx_in['data'].shape[0],vtx_in['data'].shape[0]))
	numsteps = 0
	
	for start in range(0,vtx_in['data'].shape[1]-windowSize+1,stepSize):
	
		tmp = correlation(vtx_in,start,start+windowSize)['data']
		tmp[tmp<threshold] = 0 # Threshold
		tmp[tmp>0.001] = 1 # Binarize
		
		counts += tmp
		numsteps +=1
		
	counts = counts/float(numsteps)
	
	entropy = numpy.zeros(counts.shape[0])
	
	for i,row in enumerate(counts):
	
		# Binary entropy per-entry ( p(link) vs. p(nolink) )
		evec = numpy.nan_to_num(-row*numpy.log2(row) - (1-row)*numpy.log2(1-row))
		
		# Instead of computing the entropy of the joint distribution, we're just
		# summing up the entropies of each node pair involving this node.
		# Recall that H(X,Y) \leq H(X) + H(Y) [generalizes to n variables, of course]
		# So we're generating an _upper bound_ on the true entropy of the node.
		# If we pretend that X and Y are independent, then H(X,Y) = H(X) + H(Y)
		# Are fMRI voxels statistically independent? Some are, some ain't. So it's
		# better than a loose upper bound, but worse than an equality.
		entropy[i] = numpy.sum(evec) 
	
	entropy=entropy.reshape(entropy.shape[0],1)#.transpose()
	
	# Fix this to return a packed 1-frame VTX
	return pack_vtx(entropy, vtx_in['mask'], 'VTT/corrcoef', 'Dynamic Entropy vector [channels,1]',vtx_in['funcfile'],vtx_in['maskfile'],vtx_in['maskthresh'],mnilist=vtx_in['mnilist']), counts
		
"""
# Non-negative matrix factorization!
def NMF(vtx_in, num_bases,niter=20,facfunc=pymf.NMF):
	import pymf
	
	nmf = facfunc(vtx_in['data'],num_bases=num_bases)
	nmf.factorize(niter=niter,show_progress=True)
	
	return pack_vtx(nmf.W, vtx_in['mask'], 'VTS/NNMF', 'Nonnegative matrix factorization [voxels,features]',vtx_in['funcfile'],vtx_in['maskfile'],vtx_in['maskthresh'],mnilist=vtx_in['mnilist']), nmf
"""		

		
		



def run_window(raw_data, windowSize, stepSize, outtype='glist'):
	"""Windowed analysis.
	
	:param raw_data: each row is a channel, each column a sample
	:param windowSize: size (in time points) of window over which to compute PLI
	:param stepSize: size (in time points) of steps between windows.
	:param outtype: 'glist' returns a list of graphs, otherwise return raw PLI matrices
	"""
	
	num_channels = raw_data.shape[0]
	num_samples = raw_data.shape[1]
	
	# list of graphs
	glist = []
	
	# For each window, create PLI matrix and build a graph
	for start in range(0,num_samples-windowSize+1,stepSize):
			
		# create windowed subarray
		data = raw_data[:,start:start+windowSize]

		# generate PLI matrix
		pmat = plimatrix(data)
		
		if outtype=='glist':
			# Turn it into a graph (FIXME fixed S right now)
			g = mat2graph(pmat, thresh_mat(pmat,2.5))
		
			glist.append(g)
		else:
			glist.append(pmat)
		
		
		



# --- IO ---

def save(filename, vtx):
	"""
	Save a xifti file to disk as an HDF5 file.
	
	:param filename: File name
	:param vtx: Xifti object (dictionary)
	"""
	import h5py
	import pickle
	
	f = h5py.File(filename,'w')
	
	mask = pickle.dumps(vtx['mask'],0)
	
	# Main data payload
	#f['data']=vtx['data'] 
	dset = f.create_dataset('data', maxshape=(vtx['data'].shape[0],vtx['data'].shape[1]), data=vtx['data'].astype('float32'))
	
	# Serialized (pickle) nibabel mask
	f['mask'] = mask
	
	# Attributes
	dset.attrs['type'] = vtx['type'] 
	dset.attrs['desc'] = vtx['desc']
	dset.attrs['funcfile']=vtx['funcfile']
	dset.attrs['maskfile']=vtx['maskfile']
	dset.attrs['maskthresh']=vtx['maskthresh']
	dset.attrs['mode']=vtx['mode']
	
	if vtx['mnilist'] != None:
		list_set = f.create_dataset('mnilist', maxshape=(vtx['mnilist'].shape[0],4), data=vtx['mnilist'].astype('float32'))
	
	f.close()


def load(filename):
	"""
	Load a Xifti object from a stored HDF5 file.
	
	:param filename: file name
	
	:returns: a Xifti object (dictionary)
	"""
	import h5py
	import pickle
	
	
	f = h5py.File(filename)
	
	try:
		mask = pickle.loads(f['mask'][...])
	except:
		print "Failed to load mask -- continue at own risk!"
		mask = None
	
	dset = f['data']
	
	try:
		list_set=f['mnilist']
		mnilist = list_set[...]
	except:
		mnilist = None
		
	vtx = pack_vtx(dset[...],mask,dset.attrs['type'],dset.attrs['desc'],dset.attrs['funcfile'],dset.attrs['maskfile'],dset.attrs['maskthresh'],dset.attrs['mode'],mnilist=mnilist)
		
	f.close()
	
	return vtx
	

def pack_vtx(data,mask,type,desc,funcfile,maskfile,maskthresh,mode='fMRI',mnilist=None):
	"""
	Pack data into a VTX structure
	
	:param data: A Numpy array containing the raw data
	:param mask: A nibabel nifti structure containing the mask used for the data
	:param type: One of 'VTS', 'VTT/tsmethod' for varying tsmethods
	:param desc: Plain English description of the data payload
	:param funcfile: original functional file from which the data was derived
	:param maskfile: original maskfile (now replicated in mask)
	:param mode: Neuroimaging mode: fMRI, MEG, EEG, etc.
	:param mnilist: An array of (x,y,z) MNI 152 co-ordinates; one for each row in 'data'
		
	:returns: a VTX structure
	"""
	
	vtx = {}	
	
	vtx['data'] = data
	vtx['mask'] = mask
	vtx['type'] = type
	vtx['desc'] = desc
	vtx['funcfile']=funcfile
	vtx['maskfile']=maskfile
	vtx['maskthresh']=maskthresh
	vtx['mode']=mode
	vtx['mnilist']=mnilist

	return vtx



# Testing code

#vtx=python_vts('/Users/daley/Dropbox/newJJ/S3/func/func_res.nii.gz','/Users/daley/Dropbox/newJJ/S3/segment/gm2func.nii.gz')
#vtt=load_vtx('vttsamp.vtx')
#eta=load_vtx('crap')

# Current working prototypes in:
# /Users/daley/Dropbox/newJJ/S4

