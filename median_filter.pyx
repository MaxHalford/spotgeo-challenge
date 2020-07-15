import numpy as np

cimport cython
cimport numpy as np

ctypedef np.uint_t uint
ctypedef np.uint8_t uint8


@cython.boundscheck(False)
@cython.wraparound(False)
cdef (uint8, uint) find_quantile(uint [:] hist, float q):

    cdef uint cumsum
    cdef uint total
    cdef int i
    for i in range(hist.shape[0]):
        total += hist[i]
    cdef uint8 median
    cdef uint threshold = int(total * q)

    for median in range(hist.shape[0]):
        cumsum += hist[median]
        if cumsum >= threshold:
            return median, cumsum


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef quantile_filter(uint8[:, :] img, int width, float q):

    cdef uint8 [:, :] quantiles = np.zeros_like(img)
    cdef uint8 quantile
    cdef uint cumsum

    cdef uint [:] hist = np.zeros(256, dtype=np.uint)

    cdef int row
    cdef int col
    cdef int r
    cdef int c
    cdef int removed_col
    cdef int added_col
    cdef uint8 val
    cdef uint threshold

    cdef int n_rows
    cdef int n_cols
    cdef int size

    for row in range(img.shape[0]):

        hist[:] = 0

        # Initialize the histogram with the region that surrounds the first pixel of the current row
        for r in range(max(0, row - width), min(img.shape[0], row + width + 1)):
            for c in range(width + 1):
                hist[img[r, c]] += 1

        quantile, cumsum = find_quantile(hist, q)
        quantiles[row, 0] = quantile

        for col in range(1, img.shape[1]):

            # Rollback the histogram with the values from the oldest column
            removed_col = col - width - 1
            if removed_col >= 0:
                for r in range(max(0, row - width), min(row + width + 1, img.shape[0])):
                    val = img[r, removed_col]
                    hist[val] -= 1
                    if val <= quantile:
                        cumsum -= 1

            # Update the histogram with the values from the newest column
            added_col = col + width
            if added_col < img.shape[1]:
                for r in range(max(0, row - width), min(row + width + 1, img.shape[0])):
                    val = img[r, added_col]
                    hist[val] += 1
                    if val <= quantile:
                        cumsum += 1

            # Determine how many pixels make up the current region
            n_rows = min(row + width, img.shape[0] - 1) - max(0, row - width) + 1
            n_cols = min(col + width, img.shape[1] - 1) - max(0, col - width) + 1
            size = n_rows * n_cols
            threshold = int(size * q)

            # Update the quantile

            while cumsum > threshold or hist[quantile] == 0:
                cumsum -= hist[quantile]
                quantile -= 1
            while cumsum < threshold or hist[quantile] == 0:
                quantile += 1
                cumsum += hist[quantile]

            quantiles[row, col] = quantile

    return np.asarray(quantiles)
