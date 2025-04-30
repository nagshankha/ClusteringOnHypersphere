
def gram_schmidt(X):
    
    """
    This function convert any arbitrary basis into an orthonormal basis of the 
    same span using Gram-Schmidt process
    """
    
    if not (isinstance(X, np.ndarray) and np.issubdtype(X.dtype, np.floating)):
        raise ValueError("X must be a floating numpy array")
    else:
        pass
    
    if np.shape(X) == 2:
        n_rows, n_columns = np.shape(X)
    else:
        raise ValueError("X must be a 2D array")
    
    if n_rows != n_columns:
        raise ValueError("X must be a square matrix")    
    elif np.isclose(np.linalg.det(X), 0):
        raise ValueError("X must not be a singular matrix")
    else:
        orthonormal_basis = np.ones(np.shape(X))
        
    for row in range(n_rows):
        if row == 0:
            orthonormal_basis[row] = X[row]/np.linalg.norm(X[row])
        else:
            orthonormal_basis[row] -= np.sum(np.dot(orthonormal_basis[:row], 
                                                    orthonormal_basis[row])*
                                             orthonormal_basis[:row].T, axis=1)
            orthonormal_basis[row] /= np.linalg.norm(orthonormal_basis[row])
            
    return orthonormal_basis

def NCP_sq_lattice(n, m):
    
    """
    This function list m non-collinear points on n-dimensional square lattice
    """
    
    from sympy.utilities.iterables import multiset_permutations, permute_signs
    
    num = 0
    
    arr0 = np.zeros((n,n))
    arr0[np.tril_indices(n)] = 1
    
    arr1 = []
    for i in range(n-1):
        arr1.append(arr0[i]+arr0[(i+1):])
    arr1 = np.concatenate(arr1)
    
    def func(arr0, arr1, f):
        a = []
        for r in arr0:
            a.append((f*r)+arr1)
        a = np.concatenate(a)
        a = np.unique(a, axis=0)
        return a
    
    step = 0; arr = []
    while(num<=m):
        if step == 0:
            step_arr = arr0
        elif step == 1:
            step_arr = arr1
        else:
            step_arr = func(arr0, arr1, step-1)
        for r in step_arr:
            mp = list(multiset_permutations(list(r)))[::-1]
            for mp_i in mp:
                p = list(permute_signs(mp_i))
                arr += p[:int(len(p)/2)]
                
        step = step+1
        num = len(arr)
        
    return normalize(arr[:m])