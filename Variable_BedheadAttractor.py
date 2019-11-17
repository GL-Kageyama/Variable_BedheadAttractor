#=================================================================================
#----------------    Variable Value and Bedhead Attractor    ---------------------
#=================================================================================

#-------------     X =  sin(X * Y / b) * Y + cos(a * X - Y)    -------------------
#-------------     Y =  sin(Y) / b + X                         -------------------

#=================================================================================

import numpy as np
import pandas as pd
import panel as pn
import datashader as ds
from numba import jit
from datashader import transfer_functions as tf
from colorcet import palette_n

#---------------------------------------------------------------------------------

ps = {k:p[::-1] for k, p in palette_n.items()}

pn.extension()

#---------------------------------------------------------------------------------

@jit(nopython=True)
def bedhead_trajectory(a, b, x0, y0, n):
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    
    for i in np.arange(n-1):
        x[i+1] = np.sin(x[i] * y[i] / b) * y[i] + np.cos(a * x[i] - y[i])
        y[i+1] = np.sin(y[i]) / b + x[i]
        
    return x, y

#---------------------------------------------------------------------------------

def bedhead_plot(a=1.40, b=1.14, n=1000000, colormap=ps['blues']):
    
    cvs = ds.Canvas(plot_width=500, plot_height=500)
    x, y = bedhead_trajectory(a, b, 1, 1, n)
    agg = cvs.points(pd.DataFrame({'x':x, 'y':y}), 'x', 'y')
    
    return tf.shade(agg, cmap=colormap)

#---------------------------------------------------------------------------------

pn.interact(bedhead_plot, n=(1,10000000), colormap=ps)

#---------------------------------------------------------------------------------

# The value of this attractor can be changed freely.
# Try it in the jupyter notebook.

