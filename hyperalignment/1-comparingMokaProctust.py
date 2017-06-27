import copy
import csv
import matplotlib.pyplot as plt
import MarkProcrustes
import moka
import numpy as np
import pickle
import scipy
import scipy.stats
import sklearn.decomposition
import xifti as xi


data = pickle.load(open('haxbyData.pkl'))
targets = np.array(data['targets'])
data = np.array(data['data'])

# Source and Target subjects
s = data[0].T
t = data[1].T

DO_PCA = True
if DO_PCA:
	# Calculating the top X principle components
	s = s.T
	t = t.T
	pca = sklearn.decomposition.PCA(n_components=25)
	s = pca.fit_transform(s)
	#print pca.explained_variance_ratio_, np.sum(pca.explained_variance_ratio_)
	t = pca.fit_transform(t)
	#print pca.explained_variance_ratio_, np.sum(pca.explained_variance_ratio_)
	
	# Putting back the right way
	s = s.T
	t = t.T	

else:
	# Find the top related voxels (probably not as good as PCA[?])
	top_s = moka.fast_top_voxels(s, t, 150)
	top_t = moka.fast_top_voxels(t, s, 150)
	# Updating Source and Target to be the top voxels
	s = s[top_s]
	t = t[top_t]

	#s = s.T
	#t = t.T

proj = MarkProcrustes.procrustes(s,t)
proc_s = s.dot(proj)

# Create the transform
T,b = moka.build_xform(proc_s,t)
#T,b = moka.build_xform_pseudo(s,t)
#T,b = moka.build_xform_gauss(s,t)


# Applying the transform
new_s = moka.apply_xform(proc_s, T, b)



print 't minus new_s (procrust + Russia) ', np.sum(abs(t) - abs(new_s))
print 't minus just procrust ', np.sum(abs(t) - abs(s.dot(proj)))

plt.matshow(s)
plt.colorbar()
plt.title('s')

plt.matshow(t)
plt.colorbar()
plt.title('t')

plt.matshow(new_s)
plt.colorbar()
plt.title('new_s')

plt.matshow(s.dot(proj))
plt.colorbar()
plt.title('new_s_procrust')

T,b = moka.build_xform(s,t)
moka_s = moka.apply_xform(s, T, b)
plt.matshow(moka_s)
plt.colorbar()
plt.title('moka_s')

plt.show()



