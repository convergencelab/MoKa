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

def plot_RSA(mat):
	distMat_s = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(mat.T, 'correlation'))
	plt.matshow(distMat_s)
	for i in range(0,56,8):
		plt.axvline(float(i)-0.5,color='k')
		plt.axhline(float(i)-0.5,color='k')
	plt.show()


data = pickle.load(open('haxbyData.pkl'))
targets = np.array(data['targets'])
data = np.array(data['data'])

# Sort them so the stimu is grouped together

s = data[0]
t = data[3]

s = s[np.argsort(targets[0])]
t = t[np.argsort(targets[1])]

# Source and Target subjects
s = s.T
t = t.T




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
	top_s = moka.fast_top_voxels(s, t, 250)
	top_t = moka.fast_top_voxels(t, s, 250)
	# Updating Source and Target to be the top voxels
	s = s[top_s]
	t = t[top_t]

	#s = s.T
	#t = t.T



print 'procrust'
proj = MarkProcrustes.procrustes(s,t)
proc_s = s.dot(proj)

# Create the transform
print 'moKa'
T,b = moka.build_xform(s,t)
moka_s = moka.apply_xform(s, T, b)

T,b = moka.build_xform(proc_s,t)
moka_proc_s = moka.apply_xform(proc_s, T, b)



#plot_RSA(s)
#plot_RSA(t)

ALL_STACK = np.hstack((s,t,moka_s,proc_s,moka_proc_s)).T
distMat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(ALL_STACK, 'correlation'))
plt.matshow(distMat)

distMat_s = distMat[56*0:56*1,56*0:56*1]
distMat_t = distMat[56*1:56*2,56*1:56*2]
distMat_moka_s = distMat[56*2:56*3,56*2:56*3]
distMat_proc_s = distMat[56*3:56*4,56*3:56*4]
distMat_moka_proc_s = distMat[56*4:56*5,56*4:56*5]

for i in range(0,56*5,56):
	plt.axvline(float(i)-0.5,color='k')
	plt.axhline(float(i)-0.5,color='k')

plt.show()



plt.matshow(distMat_t)
plt.colorbar()
plt.matshow(distMat_proc_s)
plt.colorbar()
plt.matshow(distMat_moka_proc_s)
plt.colorbar()

plt.matshow(distMat_t - distMat_proc_s)
print np.sum(distMat_t - distMat_proc_s)
plt.colorbar()
for i in range(0,56,8):
	plt.axvline(float(i)-0.5,color='k')
	plt.axhline(float(i)-0.5,color='k')

plt.matshow(distMat_t - distMat_moka_proc_s)
print np.sum(distMat_t - distMat_moka_proc_s)
plt.colorbar()
for i in range(0,56,8):
	plt.axvline(float(i)-0.5,color='k')
	plt.axhline(float(i)-0.5,color='k')

plt.show()



