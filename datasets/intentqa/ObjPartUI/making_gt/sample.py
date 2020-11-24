import matplotlib.pyplot as plt
import numpy as np

# two images x1 is initially visible, x2 is not
x1 = np.random.random((100, 100))
x2 = np.random.random((150, 175))

# arbitrary extent - both images must have same extent if you want
# them to be resampled into the same axes space
extent = (0, 1, 0, 1)
im1 = plt.imshow(x1, extent=extent)
im2 = plt.imshow(x2, extent=extent)
im2.set_visible(False)


def toggle_images(event):
    'toggle the visible state of the two images'
    if event.key != 't':
        return
    b1 = im1.get_visible()
    b2 = im2.get_visible()
    im1.set_visible(not b1)
    im2.set_visible(not b2)
    plt.draw()

plt.connect('key_press_event', toggle_images)

plt.show()
