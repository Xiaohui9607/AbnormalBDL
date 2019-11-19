import glob
import skimage
from skimage import io, transform, filters
from skimage import segmentation as sg
from matplotlib import pyplot as plt

images = glob.glob("../data/OCT_raw/*/*/*.jpeg")

for image_path in images:
    image = io.imread(image_path)
    image = transform.resize(image, (512, 512))
    io.imshow(image)
    plt.show()
    image = filters.gaussian(image, 2)
    # image = sg.flood_fill(image, (0, 0), 0, tolerance=10)
    # image = sg.flood_fill(image, (image.shape[0]-1, 0), 0, tolerance=10)
    # image = sg.flood_fill(image, (0, image.shape[1]-1), 0, tolerance=10)
    # image = sg.flood_fill(image, (image.shape[0]-1, image.shape[1]-1), 0, tolerance=10)
    # if height > 800:
    io.imshow(image)
    plt.show()
    # io.imsave(image_path.replace("OCT_raw", "OCT"), image)
    pass