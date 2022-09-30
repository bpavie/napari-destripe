"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
import numpy as np
import pystripe
import tqdm
import itertools
import multiprocessing
from multiprocessing import Pool
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari import Viewer
from napari.layers import Image


def worker(im, filter_bandwith1=256, filter_bandwith2=256, level=7, wavelet='db2'):
    destriped_im = pystripe.filter_streaks(im, [filter_bandwith1, filter_bandwith2], level, wavelet)
    return destriped_im


@thread_worker(progress=True)  # progress=True triggers animation of the "Activity" icon in napari
def destripe_thread(vol, filter_bandwith: int = 256, level: int = 7, wavelet: str = 'db2'):
    # If the image shape is 3, assume it is ZYX
    if len(vol.shape) == 3:
        z_number = vol.shape[0]
        # create the threadpool
        workers = multiprocessing.cpu_count()
        if z_number < workers:
            workers = z_number
        with Pool(workers) as p:
            images = [im for im in vol]
            items = list(zip(images,
                             itertools.repeat(filter_bandwith, z_number),
                             itertools.repeat(filter_bandwith, z_number),
                             itertools.repeat(level, z_number),
                             itertools.repeat(wavelet, z_number)))
            q = list(tqdm.tqdm(p.starmap(worker, items)))
        destriped_vol = np.array(q)
        return destriped_vol
    # If the image shape is 2, assume it is sYX
    elif len(vol.shape) == 2:
        destriped_vol = worker(im=vol, filter_bandwith1=filter_bandwith, filter_bandwith2=filter_bandwith, level=level,
                               wavelet=wavelet)
        return destriped_vol
    else:
        print('Too many image dimensions')
        return


# Only discrete wavelet families are supported
# List given by print(pywt.wavelist(kind='discrete'))
@magic_factory(wavelet={'choices': ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11',
                                    'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21',
                                    'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31',
                                    'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38',
                                    'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8',
                                    'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5',
                                    'bior6.8',
                                    'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9',
                                    'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17',
                                    'dmey',
                                    'haar',
                                    'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8',
                                    'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5',
                                    'rbio6.8',
                                    'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11',
                                    'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20'],
                        'label': 'Wavelet Choice:',
                        'tooltip': 'Name of the discrete mother wavelet(default is db2). See PyWavelets '
                                   'for more options'},
               auto_call=False)
def destripe(viewer: Viewer,
             image: Image,
             filter_bandwith: int = 256,
             level: int = 7,
             wavelet: str = 'db2') -> Image:
    # Accept only 2d/3d shape image data
    if len(image.data.shape) < 2 or len(image.data.shape) > 3:
        print('Support only 3D or 2D images')
        return
    image_name = image.name
    destripe_worker = destripe_thread(image.data, filter_bandwith, level, wavelet)  # create "worker" object
    destripe_worker.returned.connect(lambda l: _update_layer(viewer, l, image_name + '-' +
                                                             wavelet + '-' + str(filter_bandwith) + '-' + str(level)))
    destripe_worker.start()  # start the thread!


def _update_layer(viewer, image, layer_name):
    try:
        # if the layer exists, update the data
        viewer.layers[layer_name].data = image
    except KeyError:
        # otherwise add it to the viewer
        viewer.add_image(image, name=layer_name)
