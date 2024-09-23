import cv2
import matplotlib.pyplot as plt
import numpy as np

from cornerdetection.hessian_detector import HessianDetector
from cornerdetection.preprocessor import read_image_as_normalized_grayscale
from cornerdetection.utils import get_eigenvectors_at_maxima


def plot_eigenvectors(hessian, dot, image_ary):
    first_eigenvector_x = hessian.f_xx - hessian.f_yy + np.sqrt((hessian.f_xx - hessian.f_yy) *
                                                                (hessian.f_xx - hessian.f_yy) + 4 * hessian.f_xy ** 2)
    first_eigenvector_y = 2 * hessian.f_xy

    second_eigenvector_x = hessian.f_xx - hessian.f_yy - np.sqrt((hessian.f_xx - hessian.f_yy)
                                                                 * (
                                                                         hessian.f_xx - hessian.f_yy) + 4 * hessian.f_xy ** 2)
    second_eigenvector_y = 2 * hessian.f_xy

    first_eigenvector_at_max = get_eigenvectors_at_maxima(first_eigenvector_x, first_eigenvector_y, dot)
    second_eigenvector_at_max = get_eigenvectors_at_maxima(second_eigenvector_x, second_eigenvector_y, dot)

    fig, ax = plt.subplots(figsize=(12, 7))
    # ax.axis('equal')
    ax.imshow(image_ary, cmap='gray', vmin=0)
    ax.quiver(dot[:, 1], dot[:, 0], first_eigenvector_at_max[:, 1],
              first_eigenvector_at_max[:, 0], color='red', label="First Eigenvector", headlength=3)
    ax.quiver(dot[:, 1], dot[:, 0], second_eigenvector_at_max[:, 1],
              second_eigenvector_at_max[:, 0], color='blue', label="Second Eigenvector", headlength=3)
    ax.set_title('Hessian Eigenvectors at Maxima')
    ax.set(xticklabels=[], yticklabels=[])
    ax.set_axis_off()
    ax.legend()
    plt.show()


def plot_gradients(hessian, image_ary):
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, figsize=(24, 10))
    ax0.imshow(image_ary, cmap='gray')
    ax0.set_title('Puzzleboard Cutout')
    ax1.imshow(hessian.f_x, cmap='gray', vmin=0)
    ax1.set_title('Hessian First Derivative in x Direction')
    ax2.imshow(hessian.f_xx, cmap='gray', vmin=0)
    ax2.set_title('Hessian Second Derivative in X Direction')
    ax3.imshow(hessian.f_y, cmap='gray', vmin=0)
    ax3.set_title('Hessian First Derivative in Y Direction')
    ax4.imshow(hessian.f_yy, cmap='gray', vmin=0)
    ax4.set_title('Hessian Second Derivative in Y direction')

    plt.show()


def plot_3d_saddle_point(image_path):
    original_image = read_image_as_normalized_grayscale(image_path)

    image = cv2.GaussianBlur(original_image, (5, 5), 1)
    hd = HessianDetector(image)
    profile, mS = hd.detect_corners(image, k=1)

    plt.imshow(original_image, cmap="grey")
    plt.imshow(mS, cmap="hot", alpha=0.5)
    plt.title("Hessian Detector Response")
    plt.colorbar(label="Response intensity")
    plt.show()

    x = np.arange(0, image.shape[1], 1)
    y = np.arange(0, image.shape[0], 1)
    X, Y = np.meshgrid(x, y)

    #
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, np.zeros(image.shape),
                    facecolors=plt.cm.gray(original_image / np.max(original_image)), rstride=1, cstride=1, vmin=0,
                    shade=False)
    ax.plot_surface(X, Y, mS, alpha=0.3, vmin=mS.min(), rstride=1, cstride=1, shade=False)

    ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
    ax.set_zlabel('Response S_f')
    # ax.set_title("Hessian Detector Response \"-S\"")
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    # ax.set_axis_off()
    # plt.savefig(os.path.join("/Users/justus/Documents/uni/BA/bachelor-thesis/thesis/jbiermann_thesis/images/corner-detection", "chessboard-3d"), dpi=400)
    #
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot_surface(X, Y, np.zeros(image.shape),
    #                 facecolors=plt.cm.gray(original_image / np.max(original_image)), rstride=1, cstride=1, shade=False)
    # mS = np.where(mS < 0.03, 0, 1)
    # ax.plot_surface(X, Y, mS, alpha=0.3, vmin=mS.min(), cmap="hot", rstride=1, cstride=1, shade=False)
    #
    # ax.set(xticklabels=[], yticklabels=[])
    # ax.set_zlabel('Response -S')
    # ax.set_title("Hessian Detector Response \"-S\" after elimination")
    #
    plt.show()


if __name__ == '__main__':
    plot_3d_saddle_point("/Users/justus/Documents/uni/BA/bachelor-thesis/puzzle-boards/3by3-no-text.png")
