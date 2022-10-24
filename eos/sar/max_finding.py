import numpy as np
from scipy.optimize import least_squares

def max_2d(x):
    '''Return the coordinates of the maximum of a 2D numpy array.
    '''
    return np.unravel_index(np.argmax(x), x.shape)

def get_local_maxima(vector):
        '''
        Find all local maxima within a vector of pixels.
        Local maxima are pixels whose intensity is greater than all of their neighbors.
        This excludes points on the edge of the vector.
        
        Args:
            vector (np.array): Section of the image to search.

        Returns:
            list: Found local maxima. Each element of the list is a tuple containing:
                    tuple: the (y, x)-coords of the maximum.
                    float: the intensity of the pixel.
                the list is sorted by intensity (decreasing).
        '''
        
        out = []
        for i in range(1, vector.shape[0]-1):
            for j in range(1, vector.shape[1]-1):
                window = vector[i-1:i+2, j-1:j+2]
                if max_2d(window) == (1, 1):
                    out.append(((i, j), window[1, 1]))
        
        return sorted(out, key=lambda x: x[1], reverse=True)

def interpolate_window(image):
    '''
    Performs quadratic interpolation to find the sub-pixel maximum of a 3x3
    image.
    
    Args:
        image (np.array): section of the image to interpolate.

    Returns:
        float: x-coordinate of the interpolated maximum.
        float: y-coordinate of the interpolated maximum.
        float: Interpolated intensity of the maximum.
    '''
    discrete_max = max_2d(image)
    
    if image.shape[0] < 3 or image.shape[1] < 3:
        y_max, x_max = discrete_max
        intensity = np.max(image)
    
    else:
        # Fit a bivariate second-order polynomial to the data
        
        y = np.arange(image.shape[0])
        x = np.arange(image.shape[1])
        
        def parse_coefs(c):
            '''Parse bivariate second-order polynomial coefficients into a
            matrix which can be passed to np.polynomial.polynomial functions.
            Args:
                c (list): Coefficients to parse.
                    c must be of length 6 of the form: [A, B, C, D, E, F] where
                    P = Ax**2 + By**2 + Cx + Dy + Exy + F
            '''
            A, B, C, D, E, F = c
            return np.array([F, D, B, C, E, 0, A, 0, 0]).reshape(3, 3)

        objective_function = lambda coefs: (
            np.polynomial.polynomial.polygrid2d(y, x, parse_coefs(coefs))
            - image
        ).ravel()
            
        c = least_squares(objective_function, [1, 1, 1, 1, 1, 1], method='lm')
        c = c.x
        A, B, C, D, E, F = c

        # Now that the polynomial has been fit, we compute its maximum.
        y_max = np.clip(-(2*B*C-D*E)/(4*A*B-E**2), y.min(), y.max())
        x_max = np.clip(-(2*A*D-C*E)/(4*A*B-E**2), x.min(), x.max())
        
        if np.isnan(y_max):
            y_max = discrete_max[0]
        if np.isnan(x_max):
            x_max = discrete_max[1]
        
        intensity = np.polynomial.polynomial.polyval2d(y_max, x_max, parse_coefs(c))

    return x_max, y_max, intensity

def sub_pixel_maxima(image, y, x_left, x_right, dy, zoom_factor=1):
    '''
    Finds all local maxima in a rectangular section of an image and calculate
    their sub-pixel coordinates.

    Args:
        image (np.array): Input image.
        y (int): y-coordinate of the center of the rectangle.
        x_left (int): x-coordinate of the left edge of the rectangle.
        x_right (int): x-coordinate of the right edge of the rectangle.
        dy (int): Vertical freedom from the center of the rectangle. For
            instance, if dy=1 (and zoom_factor=1), then the height of the
            rectangle is 3.
        zoom_factor (int): Factor to scale everything by, in case the image has
            been zoomed. When greater than 1, arguments passed to this function
            must be in unscaled coordinates. The function will also return
            maxima coordinates in unscaled coordinates.
            Leave at 1 for standard max finding.

    Returns:
        float: x-coordinate of V point for a given tank, in pixels (Vx).
        float: intensity of V (i.e. roof) point for a given tank (VIntensity).
    '''
    # TODO potentially should have a better definition of the window
    y, x_left, x_right, dy = np.round(np.array([y, x_left, x_right, dy]) * zoom_factor).astype(int)
    
    # Checking that the window isn't off the side of the image
    if x_left < 0:
        print('sub_pixel_maxima warning: x_left = {} < 0!'.format(x_left))
        x_left = 0
    if x_right > image.shape[1] - 1:
        print('sub_pixel_maxima warning: x_right = {} > {}!'.format(x_right, image.shape[1]-1))
        x_right = image.shape[1] - 1
    if (x_right <= 0) or (x_left > image.shape[1]-1):
        return None, None, None
    
    x_left = int(np.clip(x_left - zoom_factor, 0, np.inf))
    x_right = x_right + zoom_factor
    
    vector = image[y-dy:y+(zoom_factor-1)+dy+1, x_left:x_right+(zoom_factor-1)+1]
    maxima = get_local_maxima(vector)
    
    x_maxima = []
    y_maxima = []
    intensities = []
    for maximum, _ in maxima:
        window = vector[
            maximum[0]-1:maximum[0]+2,
            maximum[1]-1:maximum[1]+2
        ]
        
        x_max, y_max, intensity = interpolate_window(window)
        # Now that we have the coords of the max relative to the subwindow "window", we have to get them back in image coordinates.
        # First we go from "window" coordinates to "vector" coordinates, then from "vector" to "image".
        # For the first step we use the coordinates of the window center relative to the vector (contained in "maximum").
        # For the second step we will use x_left and y.
        
        # First step
        y_max = y_max - 1 + maximum[0]
        x_max = x_max - 1 + maximum[1]
        
        # Second step
        x_max = (x_max + x_left) / zoom_factor
        y_max = (y_max + y-dy) / zoom_factor
        
        x_maxima.append(x_max)
        y_maxima.append(y_max)
        intensities.append(intensity)
    
    return x_maxima, y_maxima, intensities

