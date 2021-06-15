import numpy as np

def symmetryness(confusion_matrix):
    trans = np.transpose(confusion_matrix)
    #symmetric component
    sym = (confusion_matrix + trans)/2
    #remove the diagonal values because, that just means "attack unsuccessful" and shouldn't count towards a higher "symmetry"
    np.fill_diagonal(sym,0)
    #print(sym)
    #antisymmetric component
    skew = (confusion_matrix - trans)/2
    #print(skew)
    # difference of the 1-norms of the two matrices, normalised by their sum
    # something like the relative difference of the two ¯\_(ツ)_/¯
    order = 1
    norm_skew = np.linalg.norm(skew, ord = order)
    norm_sym = np.linalg.norm(sym, ord = order)
    symmetryness =  (norm_sym - norm_skew) / (norm_sym + norm_skew)
    return symmetryness