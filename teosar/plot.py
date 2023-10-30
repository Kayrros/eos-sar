import imageio.v2 as imageio
import io
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import hsv_to_rgb, Normalize
from matplotlib.cm import ScalarMappable


def fig2img(fig, **kwargs):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, **kwargs)
    buf.seek(0)
    img = imageio.imread(buf)
    return img


class Bounds:
    def __init__(self, minval, maxval):
        self.minval = minval
        self.maxval = maxval

    def get_bounds(self):
        return self.minval, self.maxval


class PercentileBounds(Bounds):
    def __init__(self, array, percentile=5):
        if percentile < 0:
            print("Warning: Negative percentile given, defaulting to 5")
            percentile = 5
        elif percentile > 50:
            print("Warning: Percentile greater then 50, defaulting to 5")
            percentile = 5
        minval = np.nanpercentile(array, percentile)
        maxval = np.nanpercentile(array, 100 - percentile)
        super().__init__(minval, maxval)


def to_uint8(array, bounds_provider):
    minval, maxval = bounds_provider.get_bounds()
    # linear function to map minval to 0 and maxval to 255
    normalized = (array - minval) / (maxval - minval) * 255

    normalized[normalized < 0] = 0
    normalized[normalized > 255] = 255
    return normalized.astype(np.uint8)


def phase_to_jet(image):
    # Get the color map by name:
    cm = plt.get_cmap("jet")

    # Apply the colormap like a function to any array:
    colored_image = cm(to_uint8(image, Bounds(-np.pi, np.pi)))

    # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
    # But we want to convert to RGB in uint8 and save it
    return (colored_image[:, :, :3] * 255).astype(np.uint8)


def sar_amp_to_pretty_uint8(amp, p=5):
    return to_uint8(amp, PercentileBounds(amp, p))


def save_imgs_as_gif(gif_path, images, duration=0.1):
    imageio.mimsave(gif_path, images, duration=duration)


def plot_phi(
    phi,
    cmap="jet",
    title="",
    figsize=None,
    remove_ticks=True,
    fig_out_path=None,
    **save_fig_kwargs,
):
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(phi, cmap=cmap, interpolation="nearest", vmin=-np.pi, vmax=np.pi)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax, orientation="vertical", label="Phase (rad)")

    ax.set_title(title)

    if remove_ticks:
        ax.set_yticks([])
        ax.set_xticks([])

    fig.tight_layout()

    if fig_out_path is not None:
        fig.savefig(fig_out_path, **save_fig_kwargs)

    plt.show()


def plot_amp(
    amp,
    vmin,
    vmax,
    title="",
    figsize=None,
    remove_ticks=True,
    fig_out_path=None,
    **save_fig_kwargs,
):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(amp, cmap="gray", vmin=vmin, vmax=vmax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax, orientation="vertical", label="Amplitude")

    ax.set_title(title)

    if remove_ticks:
        ax.set_yticks([])
        ax.set_xticks([])

    fig.tight_layout()

    if fig_out_path is not None:
        fig.savefig(fig_out_path, **save_fig_kwargs)

    plt.show()


def plot_amp_phi(
    amp,
    vmin,
    vmax,
    phi,
    phi_cmap="jet",
    title="",
    figsize=None,
    remove_ticks=True,
    fig_out_path=None,
    **save_fig_kwargs,
):
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    im_amp = axs[0].imshow(amp, cmap="gray", vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(axs[0])
    cax_amp = divider.append_axes("bottom", size="5%", pad=0.3)
    fig.colorbar(im_amp, cax=cax_amp, orientation="horizontal", label="Amplitude")

    im = axs[1].imshow(
        phi, cmap=phi_cmap, interpolation="nearest", vmin=-np.pi, vmax=np.pi
    )

    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    fig.colorbar(im, cax=cax, orientation="horizontal", label="Phase (rad)")

    if remove_ticks:
        for ax in axs:
            ax.set_yticks([])
            ax.set_xticks([])

    fig.suptitle(title)

    fig.tight_layout()

    if fig_out_path is not None:
        fig.savefig(fig_out_path, **save_fig_kwargs)

    plt.show()


def cmpx_interf_to_rgb(amp, vmin, vmax, phi):
    # use angle to determine hue, normalized from 0-1
    min_phi = -np.pi
    max_phi = np.pi
    h = (phi - min_phi) / (max_phi - min_phi)

    # value is set as a function of amplitude, normalized
    amp_clipped = np.clip(amp, vmin, vmax)
    v = (amp_clipped - vmin) / (vmax - vmin)

    # saturation taken as 1
    s = np.ones_like(v)

    hsv = np.stack([h, s, v], axis=2)

    c = hsv_to_rgb(hsv)

    return c


def plot_cmpx_interf_rgb(
    amp,
    vmin,
    vmax,
    phi,
    title="",
    figsize=None,
    remove_ticks=True,
    fig_out_path=None,
    **save_fig_kwargs,
):
    c = cmpx_interf_to_rgb(amp, vmin, vmax, phi)

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(c)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    mappable = ScalarMappable(Normalize(vmin=-np.pi, vmax=np.pi), cmap="hsv")

    fig.colorbar(mappable, cax=cax, orientation="vertical", label="Phase(rad)")

    ax.set_title(title)

    if remove_ticks:
        ax.set_yticks([])
        ax.set_xticks([])

    fig.tight_layout()

    if fig_out_path is not None:
        fig.savefig(fig_out_path, **save_fig_kwargs)

    plt.show()
