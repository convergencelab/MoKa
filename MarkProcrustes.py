# "Going all Procrustes" on the Haxby lab's Procrustes analysis code 

import numpy as np

def procrustes(source,target,scaling=True, reflection=True, reduction=False,
                 oblique=False, oblique_rcond=-1):

        """Procrustean transformation. Finds the best linear mapping from
           source --> target and returns the matrix encoding that mapping.

        :param source : dataset or ndarray
          Source space for determining the transformation. 
        :param target : dataset or ndarray or Null
          Target space for determining the transformation
        :param scaling : bool
          Scale data for the transformation (no longer rigid body
          transformation)
        :param reflection : bool
          Allow for the data to be reflected (so it might not be a rotation).
          Effective only for non-oblique transformations
        :param reduction : bool
          If true, it is allowed to map into lower-dimensional
          space. Forward transformation might be suboptimal then and reverse
          transformation might not recover all original variance
        :param oblique : bool
          Either to allow non-orthogonal transformation -- might heavily overfit
          the data if there is less samples than dimensions. Use `oblique_rcond`.
        :param oblique_rcond : float
          Cutoff for 'small' singular values to regularize the inverse. See
          :class:`~numpy.linalg.lstsq` for more information.
          
        :returns: Projection mapping source --> target
        
        """
        
        datas = ()
        odatas = ()
        means = ()
        shapes = ()


        for i, ds in enumerate((source, target)):
            data = ds
            
            if i == 0:
                mean = 0
            else:
                mean = data.mean(axis=0)
                
            data = data - mean
            means += (mean,)
            datas += (data,)
            shapes += (data.shape,)

        # shortcuts for sizes
        sn, sm = shapes[0]
        tn, tm = shapes[1]

        # Check the sizes
        if sn != tn:
            raise ValueError, "Data for both spaces should have the same " \
                  "number of samples. Got %d in source and %d in target space" \
                  % (sn, tn)

        # Sums of squares
        ssqs = [np.sum(d**2, axis=0) for d in datas]

        # check for being invariant?
        for i in xrange(2):
            if np.all(ssqs[i] <= np.abs((np.finfo(datas[i].dtype).eps
                                       * sn * means[i] )**2)):
                raise ValueError, "For now do not handle invariant in time datasets"

        norms = [ np.sqrt(np.sum(ssq)) for ssq in ssqs ]
        normed = [ data/norm for (data, norm) in zip(datas, norms) ]

        # add new blank dimensions to source space if needed
        if sm < tm:
            normed[0] = np.hstack( (normed[0], np.zeros((sn, tm-sm))) )

        if sm > tm:
            if reduction:
                normed[1] = np.hstack( (normed[1], np.zeros((sn, sm-tm))) )
            else:
                raise ValueError, "reduction=False, so mapping from " \
                      "higher dimensionality " \
                      "source space is not supported. Source space had %d " \
                      "while target %d dimensions (features)" % (sm, tm)

        source, target = normed
        if oblique:
            # Don't do this or FIXME make it better.
            if sn == sm and tm == 1:
                T = np.linalg.solve(source, target)
            else:
                T = np.linalg.lstsq(source, target, rcond=oblique_rcond)[0]
            ss = 1.0
        else:
            # Orthogonal transformation
            # figure out optimal rotation via SVD
            U, s, Vh = np.linalg.svd(np.dot(target.T, source),
                                    full_matrices=False)
            T = np.dot(Vh.T, U.T)

            if not reflection:
                # then we need to assure that it is only rotation
                # "recipe" from
                # http://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
                # for more and info and original references, see
                # http://dx.doi.org/10.1007%2FBF02289451
                nsv = len(s)
                s[:-1] = 1
                s[-1] = np.linalg.det(T)
                T = np.dot(U[:, :nsv] * s, Vh)

            # figure out scale and final translation
            ss = sum(s)

        # select out only relevant dimensions
        if sm != tm:
            T = T[:sm, :tm]

        scale = ss * norms[1] / norms[0]
        
        # Assign projection
        if scaling:
            proj = scale * T
        else:
            proj = T
                  
        return proj, offset_out