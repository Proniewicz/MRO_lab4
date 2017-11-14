import magni

import numpy as np
import matplotlib.pyplot as plt

# Load example
from magni.imaging import domains

img = magni.afm.io.read_mi_file('example.mi').get_buffer('Topography')[0].data
img = magni.imaging.visualisation.stretch_image(img, 1)

# Show image
magni.utils.plotting.setup_matplotlib({'figure': {'figsize': (20, 10)}})
#magni.imaging.visualisation.imshow(img)

# Set transparent background on image
plt.gcf().patch.set_alpha(0.0)
plt.gca().patch.set_alpha(0.0)

# plt.show()

# Function representing matrix multiplication of the example measurement matrix


def _measure(vec):
    return vec[::2]


# Function representing transposed matrix multiplication of the example measurement matrix
def _measure_T(vec):
    output = np.zeros((h * w, 1))
    output[::2] = vec
    return output

# Wrapping of the example measurement matrix
h, w = img.shape
# scan_length = 3000.0
# num_points = 4000
additional_measure_args = []
shape = (int(h * w / 2), h * w)

# h = 256
# w = 256
scan_length = 50000.0
num_points = 50000

image_coords = magni.imaging.measurements.spiral_sample_image(h, w, scan_length, num_points)
Phi = magni.imaging.measurements.construct_measurement_matrix(image_coords, h, w)
Psi = magni.imaging.dictionaries.get_DCT((h, w))
domain = magni.imaging.domains.MultiDomainImage(Phi, Psi)
vec = magni.imaging.mat2vec(img)
domain.image = vec

print('h = ' + str(h) + ', w = ' + str(w))

vec_2 = magni.afm.reconstruction.reconstruct(domain.measurements, Phi, Psi)
img_2 = magni.imaging.vec2mat(vec_2, (h, w))

#magni.utils.plotting.setup_matplotlib({'figure': {'figsize': (20, 10)}})
magni.imaging.visualisation.imshow(img_2)

plt.show()