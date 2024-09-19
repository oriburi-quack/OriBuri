# ####################################################################
# ####                                                               #
# ####    OriBuri (Originated From SIPAS, (c) GEO-CODING)            #
# ####                                                               #
# ####    Copyright(c) Seungjun Lee                                  #
# ####                   Yonsei Univ. (Seoul, South Korea)           #
# ####                   Department of Earth System Science          #
# ####                                                               #
# ####    Version: 1.6                                               #
# ####    last update : 2024.06.03                                   #
# ####    Since 2023.03                                              #
# ####                                                               #
# ####################################################################

# ####################################################################
#
# 2024.06.03
#
# getTif     : SLC unavailable
# save_tif   : Bug fixed
# getSubset  ; Bug fixed
# save_file  : Added
# getK5h5    : Deleted
# getProduct : Deleted
# plotall    : Deleted
# 
# ####################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider
from rasterio.windows import Window
from pyproj import Proj
from PIL import Image, ImageDraw
# rasterio to gdal
# main function

def getTif(path, file):

    '''
    getTiff(path, file)
    
    SLC Products are unavailable

    path: directory to your tiff file
    filename: file name of your tiff file including type of file (.tif etc)
    type of filename have to be list
    dtype: 'grd' for GRD files, 'slc' for slc files
    '''
    import rasterio
    
    with rasterio.open(path + file) as src:
        img = src.read()
        img_meta = src.meta
        bound = src.bounds
            
    return {'Band': np.array(img)[0],
            'Product Name': file,
            'Meta Data': img_meta, 
            'coords': (bound[0], bound[1], bound[2], bound[3])}

def longlat2window(lon, lat, dataset):
    """
    Args:
        lon (tuple): Tuple of min and max lon
        lat (tuple): Tuple of min and max lat`
        dataset: Rasterio dataset

    Returns:
        rasterio.windows.Window``
    """

    from math import floor, ceil

    p = Proj(dataset.crs)
    t = dataset.transform
    xmin, ymin = p(lon[0], lat[0])
    xmax, ymax = p(lon[1], lat[1])
    col_min, row_min = ~t * (xmin, ymin)
    col_max, row_max = ~t * (xmax, ymax)
    return Window.from_slices(rows=(floor(row_max), ceil(row_min)),
                              cols=(floor(col_min), ceil(col_max)))

def getSubset (path, file, lon, lat, platform):
    
    '''
    return subset image band and meta data of geotiff file as dictionary
    getSubset (path, file, lon, lat, dtype)
    
    path : directory to your Geotiff file
    file : your file name including '.tif'
    file must be list
    dtype: 'grd' for GRD files, 'slc' for slc files
    lon, lat must be tuple
    '''

    import rasterio

    with rasterio.open(path + file) as src:
        window = longlat2window(lon, lat, src)
        img = src.read(window = window)
        img_meta = src.meta
        p = Proj(src.crs)
        t = src.transform
        xmin, ymin = p(lon[0], lat[0])
        xmax, ymax = p(lon[1], lat[1])
        src.close()
    
        
            
    return {'Band': np.array(img)[0], 
            'Platform': platform,
            'Product Name': file,
            'Meta Data': img_meta, 
            'coords': (xmin, ymin, xmax, ymax)}
        
def tif_match (stack):

    prod = stack.copy()
    
    y = []
    x = []
    
    for i in range (len(prod)):
        a, b = np.shape(prod[i]['Band'])
        y.append(a)
        x.append(b)
        
    y = np.min(y)
    x = np.min(x)
    
    for i in range (len(prod)):
        prod[i]['Band'] = prod[i]['Band'][:y, :x]
    
    print(y,x)
    
    return prod

def save_tif(product, path):
    import rasterio
    from rasterio import Affine

    data = product.copy()
    new_coords = data['coords']
    save_path = path
    with rasterio.open(save_path, 'w',
                       driver='GTiff',
                       height=data['Band'].shape[0],
                       width=data['Band'].shape[1],
                       count=1,
                       dtype=str(data['Band'].dtype),
                       crs=data['Meta Data']['crs'], transform=Affine((new_coords[2] - new_coords[0]) / data['Band'].shape[1], 0.0, new_coords[0], 0.0, (new_coords[1] - new_coords[3]) / data['Band'].shape[0], new_coords[3])) as dst:
        dst.write(data['Band'], 1)

def save_file(product, path):
    if path[-3::] == 'npy':
        import numpy as np
        np.save(path, product['Band'])
        print('###  Product has been Saved as .npy  ###')

    if path[-3::] == 'pkl':
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(product, f)
        print('###  Product has been Saved as .pkl  ###')

def imgshow (img, clim, title):
    '''
    show input image

    img: only one image is available (2D array)
    clim: set color axis as tuple
    title: set image title
    '''

    def onclick(event):
        if event.button == 1:
            print(f'pixel coords: x={int(event.xdata)}, y={int(event.ydata)}')
            print('pixel value:', img[int(event.ydata), int(event.xdata)])

    fig, ax = plt.subplots()
    a = plt.imshow(img, cmap = 'gray')
    a.set_clim(clim)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title(title)

    return plt.show()

def imgshow_sq(stack, clim, title):

    num_images = len(stack)
    current_index = 0


    imlist = [item['Band'] for item in stack]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.15)

    image = ax.imshow(imlist[current_index], cmap='gray')
    image.set_clim(clim)
    plt.title(title)

    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor=axcolor)
    plt.title(stack[0]['Product Name'][:15])
    slider = Slider(ax_slider, 'Image', 0, num_images - 1, valinit=current_index, valstep=1)

    

    def update(val):
        index = int(slider.val)
        image.set_data(imlist[index])
        fig.canvas.draw_idle()
        plt.title(stack[index]['Product Name'][:15])

    slider.on_changed(update)

    plt.show()

def imgshow_ts (stack, clim, title):
 
    imlist = []
    for i in range (0, len(stack)):
         np.array(imlist.append(stack[i]['Band']))
 
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#000000','#162929', '#1F4F4E', '#247777', '#25A2A2', '#63D4D3', '#1ED0D0',
              '#00FFFF', '#67FFFF', '#91FFFF', '#B2FFFF', '#CEFFFF', '#E7FFFF', '#FFFFFF']
    cmap = LinearSegmentedColormap.from_list('my_cmap',colors,gamma=2)
    cmap

    #plt.figure(figsize = (15,15))
    def onclick(event):
        if event.button == 1:
            print(f'pixel coords: x={int(event.xdata)}, y={int(event.ydata)}')
            signal = getSignal(imlist, int(event.ydata), int(event.xdata))
            x = generate_x(imlist)

            plt.figure()
            plt.plot(x, signal, '.-', label = 'ts_signal')
            plt.ylim(0,20)
            plt.xlabel('Data Aqcuisition Sequence')
            plt.ylabel('Pixel Value')
            plt.legend()
            plt.title('Time Series Pixel Value')
            plt.show()

    fig, ax = plt.subplots()
    a = plt.imshow(temp_avg(imlist), cmap = 'gray')
    a.set_clim(clim)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title(title)
 
    return plt.show()

def show_result(result, background, type):

    if type == 'avg':

        img = []
        for i in range (0, len(background)):
            img.append(background[i]['Band'])

        avg = temp_avg(img)
        fig = plt.imshow(avg, cmap = 'gray')
        fig.set_clim(0,2)
        fig = plt.imshow(result['Band'], cmap = 'bwr', alpha = 0.5)
        fig.set_clim(-1, 1)

    if type == 'match':
        fig = plt.imshow(background['Band'], cmap = 'gray')
        fig.set_clim(0,2)
        fig = plt.imshow(result['Band'], cmap = 'bwr', alpha = 0.5)
        fig.set_clim(-1, 1)

    return plt.show()

def getHist (img, bins):

    '''
    return histogram of input image

    img: only one image is available
    bins: set histogram axis limit
    Ex) bins = np.arange(0.01, 1, 0.01)
    '''
    
    y, x = np.histogram(img, bins = bins)
    y = y / np.sum(y)
    plt.plot(x[0:-1], y, '.', label = 'Data')
    plt.legend()
    plt.title('Histogram', size = 20)
    return plt.show()

def getInt (data):
    
    '''
    return intensity image

    data: Single Look Complex product
    '''
    
    result = data.copy()
    intensity = abs(result['Band'])**2
    result['Band'] = intensity
    return result

def getAmp (data):

    '''
    return amplitude image
    getAmp (data)
    
    data: Single Look Complex product
    slc[0] : real number component of slc image
    slc[1] : imaginary number component of slc image
    '''

    result = data.copy
    amplitude = abs(result['Band'])
    result['Band'] = amplitude
    return result

def getdB (data):

    '''
    return decibell scale SAR image
    getdB (data)
    
    data: slc product is not available
    '''
    
    result = data.copy()
    img = result['Band']
    img[img == 0] = 0.00001
    dB = 10 * np.log10(img)
    result['Band'] = dB
    return result

def getRGB (red, green, blue):
    
    '''
    return RGB scale image
    
    getRGB (red, green blue)
    
    red: input image represented as red
    green: input image represented as green
    blue: input image represented as blue
    '''
    
    RGB = np.dstack((red, green))
    RGB = np.dstack((RGB, blue))
    
    return RGB

def calibration (data, platform):
    
    '''
    return sigma0 image of SLC image
    calibrtion (data, platform)
    
    data: complex SLC data with calibration factors
    platform: 'K5' for KOMPSAT-5, 'S1' for Sentinel-1
    '''
        
    if platform == 'K5':
        result = data.copy()
        cf = result['Calibration Factors']
        s = result['Band']
        cal = abs(s * cf['Rescaling Factor'])**2 * cf['Calibration Constant'] * np.sin(cf['Reference Incidence Angle'] * 0.017)
        result['Band'] = cal
    
    if platform == 'S1':
        result = data.copy()
        cal = result['Band'] / (result['Azimuth Resolution'] * result['Ground Range Resolution'])
        result['Band'] = cal

    return result

def makePolygon (product):
    '''
    show input image

    product: only one image is available (2D array)
    clim: set color axis as tuple
    title: set image title
    '''

    #plt.figure(figsize = (15,15))
    
    img = product['Band'].copy()

    polygon_coord = []
    def onclick(event):
        
        if event.button == 1:
            print(f'pixel coords: x={int(event.xdata)}, y={int(event.ydata)}')
            polygon_coord.append((int(event.xdata), int(event.ydata)))

    fig, ax = plt.subplots()
    a = plt.imshow(img, cmap = 'gray')
    a.set_clim(0,1)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    a,b = np.shape(img)
    img1 = Image.new('L', (b,a), 0)
    ImageDraw.Draw(img1).polygon(polygon_coord, outline=1, fill=1)
    mask = np.array(img1)
    
    plt.imshow(product['Band'], cmap = 'gray')
    plt.clim(0,1)
    plt.imshow(mask, cmap = 'bwr_r', alpha = 0.5)
    plt.clim(-1,1)
    plt.title('Polygon Area')
    plt.show()

    return polygon_coord, mask

def getENL (product, mode):
    print('')
    print('')
    print('######################################################################')
    print('')
    print('')
    print('getENL (Effective Number of Looks)')
    print('')
    print('Copyright (c) Seungjun Lee')
    print('              Sejong Univ. (Seoul, South Korea)')
    print('              Department of Energy Resource Engineering')
    print('')
    print('######################################################################')
    print('')
    print('')
    
    if mode == 'single':
        polygon_coord, mask = makePolygon(product)
        del polygon_coord

        img = product['Band'].copy()

        pix = img.flatten()
        pix = np.delete(pix, mask.flatten() == 0)

        mean = np.mean(pix)
        var = np.std(pix) ** 2
        enl = (mean ** 2) / var

    if mode == 'ts':
        polygon_coord, mask = makePolygon(product[0])
        del polygon_coord

        img = []
        for i in range (len(product)):

            tmp = product[i]['Band'].copy()
            img.append(tmp)
        img = np.array(img)

        enl = []
        for i in range (len(product)):
            pix = img[i].flatten()
            pix = np.delete(pix, mask.flatten() == 0)

            mean = (np.mean(pix))
            var = (np.std(pix) ** 2)
            enl.append((mean ** 2) / var)

        enl = np.mean(np.array(enl))
        print('')
        print('Equivalent Number of Looks :', enl)

    return enl

def getFBR (product, alpha, enl):
    '''
    returns Frozen Background Rference Image

    product: SAR time seires image stack
    alpha: 
    enl: Equivalent Numbur of Looks

    reference: 
    
    Thibault Taillade, Laetitia Thirion-Lefevre and Régis Guinvarc’h, 2020.
    Detecting Ephemeral Objects in SAR Time-Series Using Frozen Background-Based Change Detection,
    MDPI, Remote Sens. 2020, 12(11), 1720; https://doi.org/10.3390/rs12111720
    '''
    print('')
    print('')
    print('#########################################################')
    print('')
    print('Frozen Background Reference image')
    print('')
    print('reference: ')
    print('')
    print('Thibault Taillade, Laetitia Thirion-Lefevre and Régis Guinvarc’h, 2020.')
    print('Detecting Ephemeral Objects in SAR Time-Series Using Frozen Background-Based Change Detection,')
    print('MDPI, Remote Sens. 2020, 12(11), 1720; https://doi.org/10.3390/rs12111720')
    print('')
    print('#########################################################')

    from math import sqrt, gamma
    from alive_progress import alive_bar

    stack = []
    for i in range (len(product)):
        tmp = product[i]['Band'].copy()
        stack.append(tmp)
    stack = np.array(stack)

    a, b, c = np.shape(stack)
    fbr = np.zeros((b,c))
    fbr_idx = np.zeros((b,c))
    print('')
    print('')
    print('Number of Images:', a)
    print('Alpha:', alpha)
    print('')
    print('Processing ...')
    print('')
    with alive_bar(b, force_tty = True) as bar:
        for i in range (0, b):
            for j in range (0, c):

                signal = getSignal(stack, i, j)
                signal_sort = np.sort(signal)[::-1]


                while 1:

                    amp = np.std(signal_sort) / np.mean(signal_sort)
                    cv = sqrt(((gamma(enl) * gamma(enl + 1))/(gamma(enl + 0.5) ** 2)) -1)
                    th = cv + (alpha / (sqrt(len(signal_sort))+0.0000001))

                    if amp < th or len(signal_sort) == 1:
                        fbr[i,j] = np.mean(signal_sort)
                        fbr_idx[i,j] = len(signal_sort)
                        break
                    
                    else:
                        signal_sort = np.delete(signal_sort, 0)
                        continue
                    
            bar()

    return fbr, fbr_idx
#%% target detection
'''
def thresholding (img):
'''

def cfar_fft(data, alpha, win_size):
    '''
    return CFAR algorithm applied product

    data: input product
    alph: False alarm rate
    win_size: must be integer odd number
    '''
    print('')
    print('')
    print('CFAR (Constant False Alarm Rate) Detector')
    print('')
    print('type: FFT')
    print('flase Alarm Rate:', alpha)
    print('kernel size:', win_size)
    print('')
    print('')

    from alive_progress import alive_bar


    im = data.copy()
    a, b = im['Band'].shape
    target_pixel = win_size // 2 + 1
    TD = np.zeros(np.shape(im['Band']))

    with alive_bar(a - win_size, force_tty = True) as bar:
        for i in range (0, a - win_size):
            for j in range (0, b - win_size):
                win = im['Band'][i:i+win_size, j:j+win_size]
                clutter = win[:-1, 0].flatten()
                np.concatenate((clutter, win[-1, :-1].flatten()))
                np.concatenate((clutter, win[1:, -1].flatten()))
                np.concatenate((clutter, win[0, 1:].flatten()))

                bins = np.arange(0, 20, 0.01)
                y, x = np.histogram(clutter, bins = bins, range = (0, 10))
                y = y / np.sum(y)
                fft = np.fft.fft(y)
                fft[30:-30] = 0
                fitted = np.abs(np.fft.ifft(fft))
                p_value = ppf(fitted, x, 1-alpha)

                if win[target_pixel, target_pixel] > p_value:
                    TD[i+target_pixel, j+target_pixel] = 1
            bar()
    im['Band'] = TD

    return im

def cfar_db(data, alpha, win_size):
    '''
    return CFAR algorithm applied product

    data: input product
    alph: False alarm rate
    win_size: must be integer odd number
    '''
    print('')
    print('')
    print('CFAR (Constant False Alarm Rate) Detector')
    print('')
    print('type: Log-distribution Method')
    print('False Alarm Rate:', alpha)
    print('kernel size:', win_size)
    print('')
    print('')

    from alive_progress import alive_bar

    im = data.copy()
    a, b = im['Band'].shape
    target_pixel = win_size // 2 + 1
    TD = np.zeros(np.shape(im['Band']))
    print('processing ...')
    with alive_bar(a - win_size, force_tty = True) as bar:
        for i in range (0, a - win_size):
            for j in range (0, b - win_size):
                win = im['Band'][i:i+win_size, j:j+win_size]
                clutter = win[:-1, 0].flatten()
                clutter = np.concatenate((clutter, win[-1, :-1].flatten()))
                clutter = np.concatenate((clutter, win[1:, -1].flatten()))
                clutter = np.concatenate((clutter, win[0, 1:].flatten()))

                clutter = np.log10(clutter) * 10
                mean = np.mean(clutter)
                std = np.std(clutter)
                
                if alpha == 0.05:
                    p_value = mean + 2*std
                    
                if alpha == 0.01:
                    p_value = mean + 3*std
                    
                
                if np.log10(win[target_pixel, target_pixel])*10 > p_value:
                    TD[i+target_pixel, j+target_pixel] = 1
            bar()
    im['Band'] = TD

    return im

def ca_cfar(product, GUARD_CELLS, BG_CELLS, ALPHA):
    '''
    Copyright:
    https://qiita.com/harmegiddo/items/8a7e1b4b3a899a9e1f0c
    '''
    print('')
    print('')
    print('')
    print('######################################################################')
    print('')
    print('CA-CFAR (Constant False Alarm Rate) Detector')
    print('')
    print('Copyright (c):')
    print('    https://qiita.com/harmegiddo/items/8a7e1b4b3a899a9e1f0c')
    print('')
    print('')
    print('######################################################################')
    print('')
    print('')
    from alive_progress import alive_bar

    CFAR_UNITS = 1 + (GUARD_CELLS * 2) + (BG_CELLS * 2)
    

    inputImg = product.copy()
    ship = np.zeros((np.shape(inputImg['Band'])), np.uint8)


    with alive_bar(inputImg['Band'].shape[0] - CFAR_UNITS, force_tty = True) as bar:
        for i in range(inputImg['Band'].shape[0] - CFAR_UNITS):
            center_cell_x = i + BG_CELLS + GUARD_CELLS
            for j in range(inputImg['Band'].shape[1] - CFAR_UNITS):
                center_cell_y = j  + BG_CELLS + GUARD_CELLS
                average = 0
                for k in range(CFAR_UNITS):
                    for l in range(CFAR_UNITS):
                        if (k >= BG_CELLS) and (k < (CFAR_UNITS - BG_CELLS)) and (l >= BG_CELLS) and (l < (CFAR_UNITS - BG_CELLS)):
                            continue
                        average += inputImg['Band'][i + k, j + l]
                average /= (CFAR_UNITS * CFAR_UNITS) - ( ((GUARD_CELLS * 2) + 1) * ((GUARD_CELLS * 2) + 1) )

                if inputImg['Band'][center_cell_x, center_cell_y] > (average * ALPHA):
                    ship[center_cell_x, center_cell_y] = 1
            bar()

    inputImg['Band'] = ship
    return inputImg

#%% change detection

def ppf(pdf, x, p):

    i = 1
    
    while (sum(pdf[0:i]) < p):
        i = i + 1
    
    
    return x[i-1]

def ratio_cd (img_stack, img_sub, alpha, show):
    
    '''
    returns change map between img_stack time interval, and shows PDF histogram of ratio image of img_sub
    
    img_stack : 2 images are required, image have to include sub image
    img_sub : 2 images are required
    alpha : confidance(threshold), 0to 1; 0.05 means 99% confidence
    show : of show is 'True', it shows PDF histogram & the result
    '''
    print('')
    print('')
    print('Bi-temporal Change Detection')
    print('')
    print('alpha:', alpha)
    print('')

    
    img_stack[0][img_stack[0] == 0] = 0.00001
    img_stack[1][img_stack[1] == 0] = 0.00001
    img_sub[0][img_sub[0] == 0] = 0.00001
    img_sub[1][img_sub[1] == 0] = 0.00001
    
    
    ratio_train = np.array(img_sub[0]) / np.array(img_sub[1])
    bins = np.arange(0, 5, 0.01)
    y, x = np.histogram(ratio_train, bins = bins, range = (0, 5))
    y = y / np.sum(y)
    fft = np.fft.fft(y)
    fft[30:-30] = 0
    fitted = np.abs(np.fft.ifft(fft))
    #plt.plot(x[0:-1], y, '.-', label = 'Data')
    #plt.plot(x[0:-1], fitted, '-', color = 'red')
    #plt.legend()
    #plt.title('Fitted PDF', size = 20)
    #if show == 'True':
    #    plt.show()

    pv = ppf(fitted, x, alpha)
    print('%.2f' %(sum(fitted)))

    ratio1 = img_stack[0] / img_stack[1]
    ratio2 = img_stack[1] / img_stack[0]

    c_map = np.zeros(np.shape(ratio1))

    c_map[ratio1 < pv] = 1
    c_map[ratio2 < pv] = -1
    c_map[c_map == 0] = np.nan
    
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#00AAFF', '#FFFFFF', '#F05690']
    cmap = LinearSegmentedColormap.from_list('my_cmap',colors,gamma=2)
    cmap
    
    plt.figure(figsize = (15,15))
    fig = plt.imshow(img_stack[0], cmap = 'gray')
    fig.set_clim(0, 3)
    fig = plt.imshow(c_map, cmap = cmap)
    fig.set_clim(-1, 1)
    plt.axis('off')
    plt.title('Detected Change', size = 20)
    if show == 'True':
        plt.show()

    return c_map

'''
dual pol
'''
#%% speckle filtering
'''
def MultiLook (img, win):
'''
            
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

def lee_filter(data, size):
    
    '''
    this algorithm is based on site, linked below
    https://stackoverflow.com/questions/39785970/speckle-lee-filter-in-python

    return filtered image of img using Lee Filter method
    
    data : only one product is available
    size : kernel size(odd number)
    if size = 3, kernel size is 3x3
    '''
    print('')
    print('')
    print('')
    print('######################################################################')
    print('')
    print('Lee_Filter (Speckle Filtering Algorithm)')
    print('')
    print('Copyright (c):')
    print('    https://stackoverflow.com/questions/39785970/speckle-lee-filter-in-python')
    print('')
    print('')
    print('######################################################################')
    print('')
    print('')
    print('Kernel size:', size)
    print('')
    print('')
    result = data.copy()
    img = result['Band']
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    result['Band'] = img_output
    return result

def temp_avg (img):
    
    '''
    return temporal average image
    temp_avg (img)
    
    img: image stack
    '''
    mean_img = sum(img) / len(img)
    return mean_img

def getSignal(im_list, a, b):
    signal = []
    for i in range(len(im_list)):
        signal.append(im_list[i][a, b])
    return signal

def generate_x(vvlist):
    x = []
    for i in range (0, len(vvlist)):
        x.append(i)
    return x

def EMD_filter (imlist, n):
    '''
    return EMD(Empirical Mode Decomposition) Applied multi temporal images
    EMD_filter (imlist, n)
    
    imlist : multi-temporal image stack, registration must be done
    n: number of IMFs(1 to 3) to substract from original signal (which determines filtering effect)
    '''

    from alive_progress import alive_bar
    from PyEMD import EMD
    
    
    newimg = imlist
    a, b, c = np.shape(imlist)
    if n == 1:
        with alive_bar(b, force_tty = True) as bar:
            for i in range (0, b):
                for j in range (0, c):
                    
                    signal = getSignal(imlist, i, j)
                    x = generate_x(imlist)
                    emd = EMD()
                    imfs = emd.emd(np.array(signal), np.array(x))
                    denoised_signal = signal - imfs[0]
                    for k in range (0, a):
                        newimg[k][i,j] = denoised_signal[k]
                bar()
                        
    if n == 2:                    
        with alive_bar(b, force_tty = True) as bar:
            for i in range (0, b):
                for j in range (0, c):
                    
                    signal = getSignal(imlist, i, j)
                    x = generate_x(imlist)
                    emd = EMD()
                    imfs = emd.emd(np.array(signal), np.array(x))
                    denoised_signal = signal - imfs[0] - imfs[1]
                    for k in range (0, a):
                        newimg[k][i,j] = denoised_signal[k]
                bar()
                        
    if n == 3:                    
        with alive_bar(b, force_tty = True) as bar:
            for i in range (0, b):
                for j in range (0, c):
                    
                        signal = getSignal(imlist, i, j)
                        x = generate_x(imlist)
                        emd = EMD()
                        imfs = emd.emd(np.array(signal), np.array(x))
                        denoised_signal = signal - imfs[0] - imfs[1] - imfs[2]
                        for k in range (0, a):
                            newimg[k][i,j] = denoised_signal[k]
                bar()
        return newimg
