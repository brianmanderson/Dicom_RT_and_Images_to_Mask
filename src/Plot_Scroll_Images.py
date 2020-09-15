import copy
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
try:
    from ipywidgets import interactive, IntSlider
except:
    xxx = 1


class scroll_bar_class(object):
    ##  Original code provided by Tucker Netherton (@tnetherton), modified by @cecardenas, modularized by @bmanderson
    def __init__(self,numpy_images, title=None):
        self.title = title
        images = np.squeeze(numpy_images)
        if images.shape[-1] == 3:
            images = images[...,1]
        if len(images.shape) == 4:
            images = np.argmax(images,axis=-1)
        self.selected_images = sitk.GetImageFromArray(images,isVector=False)
        self.size = self.selected_images.GetSize()

    def custom_myshow1(self,img, margin=0.05, dpi=80 ):
        nda = sitk.GetArrayFromImage(img)
        if nda.ndim == 3:
            # fastest dim, either component or x
            c = nda.shape[-1]

            # the the number of components is 3 or 4 consider it an RGB image
            if not c in (3,4):
                nda = nda[nda.shape[0]//2,:,:]

        elif nda.ndim == 4:
            c = nda.shape[-1]

            if not c in (3,4):
                raise ReferenceError("Unable to show 3D-vector Image")

            # take a z-slice
            nda = nda[nda.shape[0]//2,:,:,:]

        ysize = nda.shape[1]
        xsize = nda.shape[1]

        # Make a figure big enough to accomodate an axis of xpixels by ypixels
        # as well as the ticklabels, etc...

        plt.close('all')
        plt.clf()

        figsize = ((1 + margin) * ysize / dpi)*2, (1 + margin) * xsize / dpi
        fig, ax = plt.subplots(1, 1, figsize= figsize)
        plt.subplots_adjust(left=0.25, bottom=0.25)

        t = ax.imshow(nda,cmap='gray',interpolation=None)
        ax.set_title('use scroll bar to navigate images')
        plt.show()

    def update(self,Z, view='2D'):
        if view == '2D':
            slices =[self.selected_images[:,:,Z]]
            dpi = 50
        self.custom_myshow1(sitk.Tile(slices, [3,1]),dpi=dpi, margin = 0.05)


def plot_Image_Scroll_Bar_Image(x):
    k = scroll_bar_class(x)
    interactive_plot = interactive(k.update,Z=IntSlider(min=0,max=x.shape[0]-1))
    output = interactive_plot.children[-1]
    output.layout.height = '600px'
    return interactive_plot


def plot_scroll_Image(x):
    '''
    :param x: input to view of form [rows, columns, # images]
    :return:
    '''
    if x.dtype not in ['float32','float64']:
        x = copy.deepcopy(x).astype('float32')
    if len(x.shape) > 3:
        x = np.squeeze(x)
    if len(x.shape) == 3:
        if x.shape[0] != x.shape[1]:
            x = np.transpose(x,[1,2,0])
        elif x.shape[0] == x.shape[2]:
            x = np.transpose(x, [1, 2, 0])
    fig, ax = plt.subplots(1, 1)
    if len(x.shape) == 2:
        x = np.expand_dims(x,axis=-1)
    tracker = IndexTracker(ax, x)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    return fig,tracker
    #Image is input in the form of [#images,512,512,#channels]


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = np.where((np.min(self.X,axis=(0,1))!= np.max(self.X,axis=(0,1))))[-1]
        if len(self.ind) > 0:
            self.ind = self.ind[len(self.ind)//2]
        else:
            self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind],cmap='gray')
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


if __name__ == '__main__':
    xxx = 1
