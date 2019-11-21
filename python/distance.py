import python.helper as helper
import numpy as np
from sklearn.cross_decomposition import CCA
import math

def distance(s1, s2, type="dist", x = None):
    # TODO: type check

    if(type == "dist"):
        mat_1 = np.matmul(np.matmul(s1, np.linalg.solve(np.matmul(np.transpose(s1), s1))), np.transpose(s1))
        mat_2 = np.matmul(np.matmul(s2, np.linalg.solve(np.matmul(np.transpose(s2), s2))), np.transpose(s2))
        return math.sqrt(np.sum(np.subtract(mat_1,mat_2))** 2)

    if(type == "trace"):
        mat_1 = np.matmul(np.matmul(s1, np.linalg.solve(np.matmul(np.transpose(s1), s1))), np.transpose(s1))
        mat_2 = np.matmul(np.matmul(s2, np.linalg.solve(np.matmul(np.transpose(s2), s2))), np.transpose(s2))
        return np.sum(np.diag(np.matmul(mat_1, mat_2))) / helper.n_col(s1)

    if(type == "canonical"):
        if(x == None):
            raise Exception("x must be specified if use type = 'canonical'")
        if(helper.n_col(x) != helper.n_row(s1)):
            raise Exception("Dimension of x is not correct.")

        cca = CCA(n_components=1)
        
        return np.mean(cca.fit(np.matmul(x, s1), np.matmul(x, s2)))