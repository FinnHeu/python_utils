# Statistical tools

def linregress_ci(x: np.ndarray, y: np.ndarray, n=5000, xn=100, conf=95):
    '''
    Perform a linear regression analysis and add confidence intervals by resampling the data, recomputing the regression 
    fit and determining the percentile.
    
    Input
    -----
    x: independent data
    y: dependent data
    n: number of resamples
    xn: number of x-points between min(x) and max(x)
    conf: confidence level in %
    
    Returns
    -------
    xx: new x-axis
    y_reg: linear regression fit
    ci_bottom: lower confidence band
    ci_top: upper confidence band
    
    '''
    # Preallocate arrays
    yy = np.empty((n, xn)) * np.nan
    
    # extract the minimum and maximum x values
    x_min, x_max = min(x), max(x)
    xx = np.linspace(x_min, x_max, xn)
    
    # Perform the initial regression
    reg = linregress(x, y)
    y_reg = xx * reg.slope + reg.intercept
                         
    ### 95% CIs ###
    # create array with indexes for resampling two arrays             
    ind = np.arange(len(x))
    
    for i in range(n):
        # draw new sample from the indices
        sample_inds = np.random.choice(ind, size=len(ind), replace=True)
                  
        # create new samples
        new_sample_y = y[sample_inds]
        new_sample_x = x[sample_inds]
                  
        # perform linreg
        reg_resample = linregress(new_sample_x, new_sample_y)
        yy[i,:] = xx * reg_resample.slope + reg_resample.intercept
        
    # Pick percentiles
    ci = (100 - conf) / 2
    ci_top = np.percentile(yy, 100 - ci, axis=0)
    ci_bottom = np.percentile(yy, ci, axis=0)
    
    
    return xx, y_reg, ci_bottom, ci_top
