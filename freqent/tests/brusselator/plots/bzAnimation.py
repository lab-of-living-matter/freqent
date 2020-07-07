import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib import animation

# Width, height of the image.
nx, ny = 500, 500
# Reaction parameters.
alpha, beta, gamma = 1, 1, 1
# A, B = 1, 3
# k1, k2, k3, k4 = 1, 2, 5, 1


def update(p, arr):
    """Update arr[p] to arr[q] by evolving in time."""

    # Count the average amount of each species in the 9 cells around each cell
    # by convolution with the 3x3 array m.
    # noiseStrength = 0.1
    q = (p + 1)
    s = np.zeros((3, ny, nx))
    m = np.ones((3, 3)) / 9
    for k in range(3):
        s[k] = convolve2d(arr[p, k], m, mode='same', boundary='wrap')
    # Apply the reaction equations
    arr[q, 0] = s[0] + s[0] * (alpha * s[1] - gamma * s[2])
    # arr[q, 0] = s[0] + A * k1 - B * s[0] * k2 + s[0]**2 * s[1] * k2 - s[0] * k4
    arr[q, 1] = s[1] + s[1] * (beta * s[2] - alpha * s[0])
    # arr[q, 1] = s[1] + B * s[0] * k2 - s[0]**2 * s[1] * k2
    arr[q, 2] = s[2] + s[2] * (gamma * s[0] - beta * s[1])
    # Ensure the species concentrations are kept within [0,1].
    np.clip(arr[q], 0, 1, arr[q])
    return arr


# Initialize the array with random amounts of A, B and C.
arr = np.random.random(size=(250, 3, ny, nx))

# Set up the image
fig, ax = plt.subplots()
im = ax.imshow(arr[0, 0], cmap=plt.cm.Blues_r)
ax.axis('off')


def animate(i, arr):
    """Update the image for iteration i of the Matplotlib animation."""

    arr = update(i, arr)
    im.set_array(arr[i, 0])
    return [im]


for ii in range(250 - 1):
    animate(ii, arr)
    plt.tight_layout()
    plt.show()
    # fig.savefig('/media/daniel/storage11/Dropbox/LLM_Danny/frequencySpaceDissipation/brusselator/bzAnimation/frame_{:03}.png'.format(ii))
    plt.pause(0.001)

fig.savefig('/media/daniel/storage11/Dropbox/LLM_Danny/frequencySpaceDissipation/brusselator/bzAnimation/finalFrame.pdf', format='pdf')

fig_kymo, ax_kymo = plt.subplots()
ax_kymo.pcolorfast(arr[:, 0, 250, :], cmap='Blues_r')
fig_kymo.savefig('/media/daniel/storage11/Dropbox/LLM_Danny/frequencySpaceDissipation/brusselator/bzAnimation/kymo.pdf', format='pdf')
# anim = animation.FuncAnimation(fig, animate, frames=120, interval=5,
#                                blit=False, fargs=(arr,))

# To view the animation, uncomment this line
plt.tight_layout()
plt.show()

# To save the animation as an MP4 movie, uncomment this line
# anim.save(filename='bz.mp4', fps=30)
