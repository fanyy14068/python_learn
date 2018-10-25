import numpy as np


def svd_smaple(A):
    """
    svd decomposition
    """
    U, Sigma, VT = np.linalg.svd(A)  # Sigma is a vector
    S = np.zeros(A.shape)
    r = len(Sigma)
    S[:r, :r] = np.diag(Sigma)
    A_hat = np.dot(np.dot(U, S), VT)
    return A_hat


if __name__ == '__main__':
    x = np.asarray([[1, 2, 3], [4, 5, 6]])
    print(svd_smaple(x))
