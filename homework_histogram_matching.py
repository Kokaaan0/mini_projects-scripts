import numpy as np
import cv2


def histogram_equalization(
    image_1: tuple[np.ndarray, np.ndarray], image_2: tuple[np.ndarray, np.ndarray]
):

    hist_1 = image_1[0]
    hist_2 = image_2[0]

    pix_num_1 = np.sum(hist_1)
    pix_num_2 = np.sum(hist_2)

    pdf_1 = np.zeros(hist_1.size)
    pdf_2 = np.zeros(hist_2.size)

    pdf_1 = hist_1 / pix_num_1
    pdf_2 = hist_2 / pix_num_2

    cdf_1 = np.zeros(hist_1.size)
    cdf_2 = np.zeros(hist_2.size)

    for i in range(cdf_1.size):
        if i > 0:
            cdf_1[i] = pdf_1[i] + cdf_1[i - 1]
        else:
            cdf_1[0] = pdf_1[0]

    for i in range(cdf_2.size):
        if i > 0:
            cdf_2[i] = pdf_2[i] + cdf_2[i - 1]
        else:
            cdf_2[0] = pdf_2[0]

    cdf_1 *= hist_1.size - 1
    cdf_2 *= hist_2.size - 1

    cdf_1 = np.round(cdf_1)
    cdf_2 = np.round(cdf_2)

    return cdf_1, cdf_2


def histogram_mapping(
    cdf_1: np.ndarray, cdf_2: np.ndarray, hist_1: np.ndarray, hist_2: np.ndarray
):

    _, cdf_repeating = np.unique(cdf_1, return_counts=True)

    new_hist_1 = np.zeros((cdf_1.size))

    for i in range(cdf_repeating.size):
        new_hist_1[i] += hist_1[i]
        j = i + 1

        while cdf_1[j] == cdf_1[i]:
            new_hist_1[i] += hist_1[j]
            j += 1

    cdf_new_hist = []
    for elem, count in zip(new_hist_1, cdf_repeating):
        cdf_new_hist.extend([elem] * count)

    gray_level = np.zeros(cdf_2.size)

    for i in range(cdf_2.size):
        for j in range(cdf_1.size):
            if cdf_1[j] == cdf_2[i]:
                gray_level[i] = cdf_new_hist[j]

    return gray_level


def main():

    image_1 = cv2.imread(".")
    image_2 = cv2.imread("....")

    hist_1, bin_edges_1 = np.histogram(image_1)
    hist_2, bin_edges_2 = np.histogram(image_2)

    bin_edges_1 = np.round(bin_edges_1)
    bin_edges_2 = np.round(bin_edges_2)

    cdf_1, cdf_2 = histogram_equalization((hist_1, bin_edges_1), (hist_2, bin_edges_2))

    gray_level = histogram_mapping(cdf_1, cdf_2, hist_1, hist_2)

    print(bin_edges_1, gray_level)


if __name__ == "__main__":
    main()
