"""Implementation of Newhall, Numerical representation of planetary ephemerides.

It's the JPL way of representing and interpolating orbits using Chebyshev
polynomials. Same notations are used. Additional notations are:
- K: number of samples, 9 in the JPL paper.
- S: upper-right block of C1

The following does not assume the samples are equidistant in time. It may, or
may not, be better... Usually, the best fit is performed at polynomial roots.
"""
import numpy as np


def chebpoly(order):
    """Compute the coefficients of a Chebyshev polynomial of a given order

    Parameters
    ----------
    order: int

    Returns
    -------
    coeffs: ndarray

    Notes
    -----
    Coefficient convention is higher degree to lower degree.
    """
    coeffs = np.zeros((order + 1,))
    coeffs[-1] = 1

    return np.polynomial.chebyshev.cheb2poly(coeffs)[::-1]


def chebderpoly(order):
    """Compute the coefficients of the derivative of
    a Chebyshev polynomial of a given order

    Parameters
    ----------
    order: int

    Returns
    -------
    coeffs: ndarray

    Notes
    -----
    Coefficient convention is higher degree to lower degree.
    """
    return np.polyder(chebpoly(order))


# precompute on module load polynomials
chebpolys = [chebpoly(i) for i in range(18)]
chebderpolys = [chebderpoly(i) for i in range(18)]


def matrix_T(N, samples):
    """Compute the matrix T

    Parameters
    ----------
    N: int
        Highest order of the Chebychev serie, should be < 17
    samples: ndarray
        Sample locations, usually -1, -3/4, -1/2, -1/4, 0, 1/4, 1/2, 3/4 and 1

    Returns
    -------
    T: ndarray
    """
    K = len(samples)
    T = np.zeros((2 * K, N + 1))

    for i in range(K):
        for j in range(N + 1):
            T[2 * i, j] = np.polyval(chebpolys[j], samples[K-i-1])
            T[2 * i + 1, j] = np.polyval(chebderpolys[j], samples[K-i-1])

    return T


def matrix_W(K):
    """Compute the weight matrix to give lesser weight to velocity during the
    least square resolution

    Parameters
    ----------
    K: int
        Number of samples

    Returns
    -------
    W: ndarray
    """
    W = np.zeros((2 * K, 2 * K))

    for i in range(K):
        W[2 * i, 2 * i] = 1.0
        W[2 * i + 1, 2 * i + 1] = 0.16

    return W


def matrix_S(N):
    """S is the upper right block of C1

    Parameters
    ----------
    N: int
        Highest order of the Chebychev serie, should be < 17

    Returns
    -------
    S: ndarray
    """
    S = np.zeros((N + 1, 4))

    for i in range(N + 1):
        S[i, 0] = np.polyval(chebpolys[i], 1)
        S[i, 1] = np.polyval(chebderpolys[i], 1)
        S[i, 2] = np.polyval(chebpolys[i], -1)
        S[i, 3] = np.polyval(chebderpolys[i], -1)

    return S


def matrix_C1(T, W, S):
    """Compute the matrix C1

    Parameters
    ----------
    T: ndarray
    W: ndarray
    S: ndarray

    Returns
    -------
    C1: ndarray
    """
    return np.block([[T.T @ W @ T, S],
                     [S.T, np.zeros((4, 4))]])


def matrix_C2(T, W, K):
    """Compute the matrix C2

    Parameters
    ----------
    T: ndarray
    W: ndarray
    K: int
        Number of samples

    Returns
    -------
    C2: ndarray
    """
    U = np.zeros((4, 2 * K))

    U[0, 0] = 1
    U[1, 1] = 1
    U[-2, -2] = 1
    U[-1, -1] = 1

    return np.block([[T.T @ W],
                     [U]])


def matrix_f(state_vectors):
    """Compute the matrix f

    Parameters
    ----------
    state_vectors: Iterable of dict

    Returns
    -------
    f: ndarray
    samples: ndarray
        Samples in domain [-1, 1]
    domain: tuple of float
        Actual time domain
    """
    K = len(state_vectors)

    f = np.zeros((2 * K, 3))

    t0 = state_vectors[0]['time']
    t1 = state_vectors[-1]['time']

    delta_t = t1 - t0

    samples = np.zeros((K,))

    for k, i in enumerate(reversed(range(K))):
        f[2 * k, :] = np.array(state_vectors[i]['position'])
        f[2 * k + 1, :] = np.array(state_vectors[i]['velocity']) * delta_t / 2

        samples[i] = -1 + 2 * (state_vectors[i]['time'] - t0) / (t1 - t0)

    return f, samples, (t0, t1)


def matrix_A(C, f, N):
    """Compute the matrix of coefficients a_n

    Parameters
    ----------
    C: ndarray
    f: ndarray
    N: int
        Highest order of the Chebychev serie, should be < 17

    Returns
    -------
    A: ndarray
    """
    A = np.zeros((N + 1, 3))

    for i in range(N + 1):
        A[i, :] = C[i:i + 1, ...] @ f

    return A


def poly_B(A, N):
    """Create the interpolation polynomial

    Parameters
    ----------
    A: ndarray
        Matrix of a_n
    N: int
        Highest order of the Chebychev serie, should be < 17

    Returns
    -------
    coeffs: ndarray

    Notes
    -----
    Convention is lower degree to higher degree; coeffs[n, ...]
    is the 3D coefficient of the term of order n.
    """
    coeffs = np.zeros((N + 1, 3))

    for i in range(N + 1):
        a_n = A[i, :][..., np.newaxis]
        coeffs_n = chebpolys[i][::-1][np.newaxis, ...]

        coeffs[:i+1, :] += (a_n @ coeffs_n).T

    return coeffs


def build_cheb_interp(state_vectors, N):
    """

    Parameters
    ----------
    state_vectors: Iterable of dict
        A state vector is a dictionary with three keys:
        - time (float) in s
        - position (list, tuple or ndarray of floats) in m
        - velocity (list, tuple or ndarray of floats) in m
        The state vectors shall be ordered w.r.t. time.
    N: int
        Highest degree of the Chebychev serie, should be < 17

    Returns
    -------
    coeffs: ndarray
    domain: tuple of float
        Time domain
    """
    f, samples, domain = matrix_f(state_vectors)

    K = len(samples)

    T = matrix_T(N, samples)
    W = matrix_W(K)
    S = matrix_S(N)

    C1 = matrix_C1(T, W, S)
    C2 = matrix_C2(T, W, K)

    C = np.linalg.inv(C1) @ C2

    A = matrix_A(C, f, N)

    coeffs = poly_B(A, N)

    return coeffs, domain


def evaluate_cheb_interp(t, coeffs, domain):
    """Evaluate the interpolation on positions

    Parameters
    ----------
    t: ndarray
    coeffs: ndarray
        Matrix of a_n
    domain: tuple of float

    Returns
    -------

    """
    t0, t1 = domain

    return np.polynomial.polynomial.polyval(
        -1 + 2 * (t - t0) / (t1 - t0),
        coeffs,
        tensor=True
    ).T

def get_diff_coeffs(coeffs, domain, der):
    """Compute the derivative coeffs
    used for speed and acceleration computation

    Parameters
    ----------
    coeffs: ndarray
        Matrix of a_n
    domain: tuple of float
    der: int 
        Order of the derivative 
    Returns
    -------
    Coefficients for the polynomial of the derivative
    """
    t0, t1 = domain
    return np.polynomial.polynomial.polyder(
        coeffs, m=der, scl=2/(t1 - t0))

    