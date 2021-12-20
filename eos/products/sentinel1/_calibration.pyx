# cython: language_level=3
# cython: infer_types=True
# cython: cxx=True

cimport cython
from libc.math cimport sqrt, abs
cimport numpy as np
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def bilinear_interpolation(tuple window, np.int32_t[::1] lines, np.int32_t[::1] cols, np.float32_t[:,::1] values):
    assert list(cols) == list(sorted(cols))
    assert list(lines) == list(sorted(lines))

    cdef int w
    cdef int h

    cdef int oriy
    cdef int orix
    orix, oriy, w, h = window
    cdef int endy = oriy + h
    cdef int endx = orix + w

    cdef np.float32_t[:,::1] result = np.zeros((h, w), dtype=np.float32)

    assert endy <= lines[len(lines) - 1] + 1
    assert endx <= cols[len(cols) - 1] + 1

    cdef int l
    cdef int c
    cdef int by
    cdef int bx
    cdef int nby
    cdef int nbx
    cdef int xx
    cdef int yy

    cdef float w11
    cdef float w12
    cdef float w21
    cdef float w22
    cdef float v11
    cdef float v12
    cdef float v21
    cdef float v22
    cdef float r

    for l in range(len(lines) - 1):
        by = lines[l]
        if by > endy:
            continue
        nby = lines[l + 1]
        if nby < oriy:
            continue

        for c in range(len(cols) - 1):
            bx = cols[c]
            if bx > endx:
                continue
            nbx = cols[c + 1]
            if nbx < orix:
                continue

            v11 = values[l+0][c+0]
            v12 = values[l+0][c+1]
            v21 = values[l+1][c+0]
            v22 = values[l+1][c+1]
            r = 1. / ((nbx - bx) * (nby - by))
            for yy in range(max(by, oriy), min(endy, nby + 1)):
                for xx in range(max(bx, orix), min(endx, nbx + 1)):
                    w11 = (nbx - xx) * (nby - yy) * r
                    w12 = (xx - bx) * (nby - yy) * r
                    w21 = (nbx - xx) * (yy - by) * r
                    w22 = (xx - bx) * (yy - by) * r
                    result[yy - oriy, xx - orix] = w11 * v11 + w12 * v12 + w21 * v21 + w22 * v22

    return np.array(result)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def apply_radiometric_calibration_float32(np.float32_t[:,::1] img, np.float32_t[:,::1] calib_coeffs, np.float32_t[:,::1] noise_coeffs, bint dont_clip_noise):
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]

    cdef np.float32_t* pimg = &img[0,0]
    cdef np.float32_t* pcalib = &calib_coeffs[0,0]
    cdef np.float32_t* pnoise
    cdef np.float32_t val

    if noise_coeffs is not None:
        pnoise = &noise_coeffs[0,0]
        if dont_clip_noise:
            for i in range(w * h):
                val = pimg[i] ** 2 - pnoise[i]
                if val < 0:
                    val = pimg[i] ** 2
                pimg[i] = val / (pcalib[i] ** 2)
        else:
            for i in range(w * h):
                pimg[i] = max(0, (pimg[i] ** 2 - pnoise[i]) / (pcalib[i] ** 2))
    else:
        for i in range(w * h):
            pimg[i] = (pimg[i] / pcalib[i]) ** 2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def apply_radiometric_calibration_complex64(np.complex64_t[:,::1] img, np.float32_t[:,::1] calib_coeffs, np.float32_t[:,::1] noise_coeffs, bint dont_clip_noise):
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]

    cdef np.complex64_t* pimg = &img[0,0]
    cdef np.float32_t* pcalib = &calib_coeffs[0,0]
    cdef np.float32_t* pnoise
    cdef np.float32_t amp
    cdef np.float32_t namp

    if noise_coeffs is not None:
        pnoise = &noise_coeffs[0,0]
        if dont_clip_noise:
            for i in range(w * h):
                amp = abs(pimg[i])
                namp = amp ** 2 - pnoise[i]
                if namp < 0:
                    namp = amp ** 2
                namp = namp / (pcalib[i] ** 2)
                pimg[i] = pimg[i] * (namp / max(amp, 1e-10))
        else:
            for i in range(w * h):
                amp = abs(pimg[i])
                namp = max(0, (amp ** 2 - pnoise[i]) / (pcalib[i] ** 2))
                pimg[i] = pimg[i] * (namp / max(amp, 1e-10))
    else:
        for i in range(w * h):
            amp = abs(pimg[i])
            namp = (amp / pcalib[i]) ** 2
            pimg[i] = pimg[i] * (namp / max(amp, 1e-10))

