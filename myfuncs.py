import os
import xks
import dask
import cftime
import shutil
import zipfile
import itertools
import numpy as np
import pandas as pd
import xarray as xr
import dask.bag as db
import xskillscore as xs


# =======================================================================
# I/O
# =======================================================================

def open_zarr(path, variables=None, region=None, preprocess=None, open_zarr_kwargs={}):
    """ Open variables from a zarr collection. Varaibles that don't exist are ignored
        without warning
    """
    def _get_variables(ds):
        """Return variables that are in dataset"""
        return ds[list(set(ds.data_vars) & set(variables))]
    try:
        ds = xr.open_zarr(path, consolidated=True, **open_zarr_kwargs)
    except KeyError:
        # Try zip file
        ds = xr.open_zarr(
            f'{path}{os.path.extsep}zip', consolidated=True, **open_zarr_kwargs)
    if preprocess is not None:
        ds = preprocess(ds) 
    if variables is not None:
        ds = _get_variables(ds)
    if region is not None:
        ds = get_region(ds, region)
    return ds


def open_zarr_forecasts(paths, variables=None, region=None, preprocess=None, convert_time_to_lead=True,
                        time_name='time', open_zarr_kwargs={}):
    """ Open multiple forecast zarr collections and stack by initial date and lead time"""
    datasets = []; aux_coords = []
    for path in paths:
        ds = open_zarr(path, variables, region, preprocess, open_zarr_kwargs)
        init_date = ds[time_name].values[0]
        lead_time = range(len(ds[time_name]))
        if convert_time_to_lead:
            aux_coords.append(
                ds[time_name].rename({time_name: 'lead_time'}
                ).assign_coords({'lead_time': lead_time}))
            datasets.append(
                ds.rename({time_name: 'lead_time'}
                ).assign_coords({'lead_time': lead_time,
                                 'init_date': init_date}))
        else:
            aux_coords.append(
                xr.DataArray(lead_time,
                            coords={time_name: ds[time_name]}).assign_coords(
                    {'init_date': init_date}))
            datasets.append(
                ds.assign_coords({'init_date': init_date}))
    if convert_time_to_lead:
        aux_coord = xr.concat(aux_coords, dim='init_date')
        dataset = xr.concat(datasets, dim='init_date')
        return dataset.assign_coords({time_name: aux_coord})
    else:
        # Broadcasting times when stacking by init_date generates large chunks so stack manually
        def _pad(ds, times):
            """ Pad with nans to fill out time dimension """
            times_pad = times[~np.in1d(times, ds.time)]
            template = ds.isel(time=0, drop=True).expand_dims(
                time=times_pad)
            if ds.chunks is not None:
                template = template.chunk({'time': ds.chunks['time'][0]})
            padding = xr.full_like(template, fill_value=np.nan, dtype=np.float32)
            return xr.concat([ds, padding], dim='time')
        times = np.sort(np.unique(np.concatenate([d.time for d in datasets])))
        times = xr.DataArray(times, dims=['time'], coords={'time': times})
        aux_coord = xr.concat([_pad(ds, times) for ds in aux_coords], dim='init_date')
        dataset = xr.concat([_pad(ds, times) for ds in datasets], dim='init_date')
        return dataset.assign_coords({'lead_time': aux_coord})


def to_zarr(ds, filename, zip=True, clobber=False):
    """ Write to zarr file and read back """
    def _zip_zarr(zarr_filename):
        """ Zip a zarr collection"""
        filename = f'{zarr_filename}{os.path.extsep}zip'
        with zipfile.ZipFile(
            filename, "w", 
            compression=zipfile.ZIP_STORED, 
            allowZip64=True) as fh:
            for root, _, filenames in os.walk(zarr_filename):
                for each_filename in filenames:
                    each_filename = os.path.join(root, each_filename)
                    fh.write(
                        each_filename,
                        os.path.relpath(each_filename, zarr_filename))
    if isinstance(ds, xr.DataArray):
        is_DataArray = True
        name = ds.name
        ds = ds.to_dataset(name=name)
    else:
        is_DataArray = False
    for var in ds.variables:
        ds[var].encoding = {}
    if zip:
        filename_open = f'{filename}{os.path.extsep}zip'
    else:
        filename_open = filename
    if (not os.path.exists(filename_open)) | clobber==True:
        ds.to_zarr(filename, mode='w', compute=True, consolidated=True)
        if zip:
            _zip_zarr(filename)
            shutil.rmtree(filename)
    ds = xr.open_zarr(filename_open, consolidated=True)
    return ds[name] if is_DataArray else ds


# =======================================================================
# Calculation of indices
# =======================================================================

def calc_drought_factor(precip_20day, dim):
    return (-10 * (precip_20day - 
                   precip_20day.min(dim)) / 
                  (precip_20day.max(dim) - 
                   precip_20day.min(dim)) + 10).rename('drought_factor')


def calc_FFDI(D, T, H, W):
    """ Calculate the Forest Fire Danger Index as in Richardson 2021
        D is the drought factor computed from the 20 day accumulated total precipitation as
            D = -10 * (P20 - min(P20)) / (max(P20) - min(P20)) + 10
            where the min and max are over time
        T is the daily maximum 2m temperature 
        H is the daily maximum 2m relative humidity
        W is the daily maximum 10m wind speed
    """
    return (( D ** 0.987 ) * np.exp( 0.0338 * T - 0.0345 * H + 0.0234 * W + 0.243147 )).rename('FFDI')


def calc_nino34(sst_anom, area=None, lat_dim='lat', lon_dim='lon'):
    """ Return Nino 3.4 index """
    box = [-5.0, 5.0, 190.0, 240.0]
        
    return lat_lon_average(sst_anom, box, area, lat_dim, lon_dim)


def calc_dmi(sst_anom, area=None, lat_dim='lat', lon_dim='lon'):
    """ Return DMI index """
    boxW = [-10.0,10.0,50.0,70.0]
    boxE = [-10.0,0.0,90.0,110.0]
    
    da_W = lat_lon_average(sst_anom, boxW, area, lat_dim, lon_dim)
    da_E = lat_lon_average(sst_anom, boxE, area, lat_dim, lon_dim)
    
    return (da_W - da_E)


def calc_sam(slp, clim_period, slp_for_clim=None,
             lat_dim='lat', lon_dim='lon', groupby_dim='time'):
    """ Returns southern annular mode index as defined by Gong, D. and Wang, S., 1999
        If you wish to use a different dataset to normalise the index, provide this as slp_for_clim
    """
    def _normalise(group, clim_group):
        """ Return the anomalies normalize by their standard deviation """
        month = group[groupby_dim].dt.month.values[0]
        months, _ = zip(*list(clim_group))
        clim_group_month = list(clim_group)[months.index(month)][1]
        clim_over = [groupby_dim, 'ensemble'] if 'ensemble' in group.dims else groupby_dim
        return (group - clim_group_month.mean(clim_over)) / clim_group_month.std(clim_over)

    slp_40 = slp.interp({lat_dim: -40}).mean(lon_dim)
    slp_65 = slp.interp({lat_dim: -65}).mean(lon_dim)
    
    clim_period_ = pd.date_range(clim_period.start, clim_period.stop, periods=2)
    period_start = cftime.datetime(
        clim_period_[0].year, clim_period_[0].month, clim_period_[0].day)
    period_end = cftime.datetime(
        clim_period_[1].year, clim_period_[1].month, clim_period_[1].day)
    
    if slp_for_clim is not None:
        period_mask = (slp_for_clim['time'] >= period_start) & (slp_for_clim['time'] <= period_end)
        slp_40_for_clim = slp_for_clim.interp(
            {lat_dim: -40}).mean(lon_dim).where(period_mask, drop=True)
        slp_65_for_clim = slp_for_clim.interp(
            {lat_dim: -65}).mean(lon_dim).where(period_mask, drop=True)
    else:
        period_mask = (slp['time'] >= period_start) & (slp['time'] <= period_end)
        slp_40_for_clim = slp_40.where(period_mask, drop=True)
        slp_65_for_clim = slp_65.where(period_mask, drop=True)

    slp_40_group = slp_40.groupby(groupby_dim+'.month')
    slp_40_group_clim = slp_40_for_clim.groupby(groupby_dim+'.month')
    slp_65_group = slp_65.groupby(groupby_dim+'.month')
    slp_65_group_clim = slp_65_for_clim.groupby(groupby_dim+'.month')

    norm_40 = slp_40_group.map(_normalise, clim_group=slp_40_group_clim)
    norm_65 = slp_65_group.map(_normalise, clim_group=slp_65_group_clim)
    
    return norm_40 - norm_65


# =======================================================================
# Forecast tools
# =======================================================================

def forecast_clim(fcst):
    """ Get the lead-dependent forecast model climatology 
    """
    def _forecast_clim(ds):
        stack_dim = [d for d in ds.dims if 'stacked_' in d][0]
        ds = ds.copy().assign_coords(
            {stack_dim: ds[stack_dim].init_date})
        return ds.groupby(f'{stack_dim}.month').mean(stack_dim)
    
    return fcst.groupby(fcst.lead_time).apply(_forecast_clim)


def get_bias(fcst, obsv):
    """ Get the lead-dependent model bias relative to the observed mean value 
    """
    fcst_clim = forecast_clim(fcst)
    obsv_clim = obsv.mean()

    return fcst_clim - obsv_clim


def remove_bias(fcst, bias):
    """ Remove the lead-dependent model bias """
    def _remove_bias(ds):
        stack_dim = [d for d in ds.dims if 'stacked_' in d][0]
        stack_coord = ds[stack_dim]
        ds = ds.copy().assign_coords(
            {stack_dim: ds[stack_dim].init_date})
        lead = np.unique(ds.lead_time)
        assert len(lead) == 1
        lead = lead[0]
        bc = ds.groupby(f'{stack_dim}.month') - bias.sel(lead_time=lead)
        return bc.assign_coords({stack_dim: stack_coord})
    
    return fcst.groupby(fcst.lead_time).apply(_remove_bias).drop('month')


def reindex_forecast(ds, init_date_dim='init_date', lead_time_dim='lead_time', time_dim='time', dropna=False):
    """ Restack by time (lead_time) a forecast dataset stacked by lead_time (time) 
        Only works on DataArrays at the moment
    """
    if lead_time_dim in ds.dims:
        index_dim = lead_time_dim
        reindex_dim = time_dim
    elif time_dim in ds.dims:
        index_dim = time_dim
        reindex_dim = lead_time_dim
    else:
        raise ValueError("Neither a time nor lead_time dimension can be found")
    swap = {index_dim: reindex_dim}
    
    reindex_coord = np.sort(np.unique(ds[reindex_dim]))
    reindex_coord = xr.DataArray(
        reindex_coord, dims=[reindex_dim], coords={reindex_dim: reindex_coord})
    reindex_coord = reindex_coord[reindex_coord.notnull()] # Using "where" here cast datetime dtypes badly
    
    def _pad(ds, reindex):
        """ Pad with nans to fill out reindex dimension """
        reindex_pad = reindex[~np.in1d(reindex, ds[reindex_dim])]
        if len(reindex_pad) > 0:
            template = ds.isel(
                {reindex_dim: len(reindex_pad)*[0]}, drop=True).assign_coords(
                {reindex_dim: reindex_pad})
            if ds.chunks is not None:
                ax = ds.get_axis_num(reindex_dim)
                template = template.chunk({reindex_dim: ds.chunks[ax][0]})
            padding = xr.full_like(template, np.nan, ds.dtype)
            # Force index dim to be nans where padded
            padding = padding.assign_coords(
                {index_dim: xr.full_like(template[index_dim].compute(), np.nan, ds.dtype)})
            return xr.concat([ds, padding], dim=reindex_dim)
        else:
            return ds

    to_concat = []
    for init_date in ds[init_date_dim]:
        fcst = ds.sel({init_date_dim: init_date})
        fcst = fcst.where(fcst[reindex_dim].notnull(), drop=True)
        # Broadcasting times when stacking by init_date generates large chunks so stack manually
        fcst_padded = _pad(fcst.swap_dims(swap), reindex_coord)
        to_concat.append(fcst_padded)
     
    concat = xr.concat(to_concat, dim=init_date_dim)
    if dropna:
        return concat.where(concat.notnull(), drop=True)
    else:
        return concat
    
    
def stack_super_ensemble(ds, by_lead):
    """ Stack into super ensemble """
    def _stack(ds, sample_dims):
        for dim in sample_dims:
            ds = ds.assign_coords({dim: range(len(ds[dim]))})
        ds_stack = ds.stack(sample=sample_dims).dropna('sample').drop('lead_time')
        return ds_stack.assign_coords({'sample': range(len(ds_stack['sample']))})
        
    if by_lead:
        sample_dims = ['stacked_init_date_time', 'ensemble']
        return ds.groupby(ds.lead_time).map(
            _stack, sample_dims=sample_dims)
    else:
        sample_dims = ['init_date', 'time', 'ensemble']
        return _stack(ds, sample_dims)


# =======================================================================
# Statistical testing
# =======================================================================

def mean_correlation_ensemble_combinations(ds, dim='init_date', ensemble_dim='ensemble'):
    """ Compute all combinations of correlations between ensemble 
        members and return the mean
    """
    combinations = np.array(
        list(itertools.combinations(range(len(ds[ensemble_dim])), 2)))
    e1 = ds.isel(
        ensemble=combinations[:,0]).assign_coords(
        {ensemble_dim: range(combinations.shape[0])})
    e2 = ds.isel(
        ensemble=combinations[:,1]).assign_coords(
        {ensemble_dim: range(combinations.shape[0])})
    corr_combinations = xs.spearman_r(e1, e2, dim=dim, skipna=True)
    mean_corr = corr_combinations.mean(ensemble_dim)
    return mean_corr


def random_resample(*args, samples,
                    function=None, function_kwargs=None, bundle_args=True,
                    replace=True):
    """
        Randomly resample from provided xarray args and return the results of the subsampled dataset passed through \
        a provided function
                
        Parameters
        ----------
        *args : xarray DataArray or Dataset
            Objects containing data to be resampled. The coordinates of the first object are used for resampling and the \
            same resampling is applied to all objects
        samples : dictionary
            Dictionary containing the dimensions to subsample, the number of samples and the continuous block size \
            within the sample. Of the form {'dim1': (n_samples, block_size), 'dim2': (n_samples, block_size)}. The first \
            object in args must contain all dimensions listed in samples, but subsequent objects need not.
        function : function object, optional
            Function to reduced the subsampled data
        function_kwargs : dictionary, optional
            Keyword arguments to provide to function
        bundle_args : boolean, optional
            If True, pass all resampled objects to function together, otherwise pass each object through function \
            separately
        replace : boolean, optional
            Whether the sample is with or without replacement
                
        Returns
        -------
        sample : xarray DataArray or Dataset
            Array containing the results of passing the subsampled data through function
    """
    samples_spec = samples.copy() # copy because use pop below
    args_sub = [obj.copy() for obj in args]
    dim_block_1 = [d for d, s in samples_spec.items() if s[1] == 1]

    # Do all dimensions with block_size = 1 together
    samples_block_1 = { dim: samples_spec.pop(dim) for dim in dim_block_1 }
    random_samples = {dim: 
                      np.random.choice(
                          len(args_sub[0][dim]),
                          size=n,
                          replace=replace)
                      for dim, (n, _) in samples_block_1.items()}
    args_sub = [obj.isel(
        {dim: random_samples[dim] 
         for dim in (set(random_samples.keys()) & set(obj.dims))}) for obj in args_sub]

    # Do any remaining dimensions
    for dim, (n, block_size) in samples_spec.items():
        n_blocks = int(n / block_size)
        random_samples = [slice(x,x+block_size) 
                          for x in np.random.choice(
                              len(args_sub[0][dim])-block_size+1, 
                              size=n_blocks,
                              replace=replace)]
        args_sub = [xr.concat([obj.isel({dim: random_sample}) 
                               for random_sample in random_samples],
                              dim=dim) 
                       if dim in obj.dims else obj 
                       for obj in args_sub]

    if function:
        if bundle_args:
            if function_kwargs is not None:
                res = function(*args_sub, **function_kwargs)
            else:
                res = function(*args_sub)
        else:
            if function_kwargs is not None:
                res = tuple([function(obj, **function_kwargs) for obj in args_sub])
            else:
                res = tuple([function(obj) for obj in args_sub])
    else:
        res = tuple(args_sub,)

    if isinstance(res, tuple):
        if len(res) == 1:
            return res[0]
    else:
        return res
    
    
def n_random_resamples(*args, samples, n_repeats, 
                       function=None, function_kwargs=None, bundle_args=True, 
                       replace=True, with_dask=True):
    """
        Repeatedly randomly resample from provided xarray objects and return the results of the subsampled dataset passed \
        through a provided function
                
        Parameters
        ----------
        args : xarray DataArray or Dataset
            Objects containing data to be resampled. The coordinates of the first object are used for resampling and the \
            same resampling is applied to all objects
        samples : dictionary
            Dictionary containing the dimensions to subsample, the number of samples and the continuous block size \
            within the sample. Of the form {'dim1': (n_samples, block_size), 'dim2': (n_samples, block_size)}
        n_repeats : int
            Number of times to repeat the resampling process
        function : function object, optional
            Function to reduced the subsampled data
        function_kwargs : dictionary, optional
            Keyword arguments to provide to function
        replace : boolean, optional
            Whether the sample is with or without replacement
        bundle_args : boolean, optional
            If True, pass all resampled objects to function together, otherwise pass each object through function \
            separately
        with_dask : boolean, optional
            If True, use dask to parallelize across n_repeats using dask.delayed
                
        Returns
        -------
        sample : xarray DataArray or Dataset
            Array containing the results of passing the subsampled data through function
    """

    if with_dask & (n_repeats > 500):
        n_args = itertools.repeat(args[0], times=n_repeats)
        b = db.from_sequence(n_args, npartitions=100)
        rs_list = b.map(random_resample, *(args[1:]), 
                        **{'samples':samples, 'function':function, 
                           'function_kwargs':function_kwargs, 'replace':replace}).compute()
    else:              
        resample_ = dask.delayed(random_resample) if with_dask else random_resample
        rs_list = [resample_(*args,
                             samples=samples,
                             function=function,
                             function_kwargs=function_kwargs,
                             bundle_args=bundle_args,
                             replace=replace) for _ in range(n_repeats)] 
        if with_dask:
            rs_list = dask.compute(rs_list)[0]
            
    if all(isinstance(r, tuple) for r in rs_list):
        return tuple([xr.concat([r.unify_chunks() for r in rs], dim='k') for rs in zip(*rs_list)])
    else:
        return xr.concat([r.unify_chunks() for r in rs_list], dim='k')
    
    
def fidelity_KS_univariate(fcst, obsv, max_period, by_lead):
    """ Perform a Kolmogorov-Smirnov test on univariate data """
    def _get_KS_statistic(fcst_ds, obsv_ds, by_lead=False):
        if by_lead:
            # Applied per lead_time within groupby
            stack_dim = [d for d in fcst_ds.dims if 'stacked_' in d][0]
            fcst_ds = fcst_ds.assign_coords({stack_dim: fcst_ds[stack_dim].time})
            fcst_ds = fcst_ds.rename({stack_dim: 'time'})
            sample_dims = ['ensemble','time']
        else:
            sample_dims = ['ensemble','init_date','time']

        # Adjust period to range of available data
        min_overlap_year = max(
            fcst_ds.time.dt.year.min().item(),
            obsv_ds.time.dt.year.min().item())
        max_overlap_year = min(
            fcst_ds.time.dt.year.max().item(),
            obsv_ds.time.dt.year.max().item())
        period = slice(str(min_overlap_year), str(max_overlap_year))

        fcst_ds_samples = fcst_ds.sel(time=period).stack(
            sample=sample_dims).dropna('sample')
        obsv_ds_samples = obsv_ds.sel(time=period).rename({'time': 'sample'})

        K, p = xks.ks1d2s(obsv_ds_samples, fcst_ds_samples, 'sample')

        if isinstance(K, xr.Dataset):
            data_vars = K.data_vars
            K = K.rename({ d: f'{d}_K_obs' for d in data_vars })
            p = p.rename({ d: f'{d}_p-value' for d in data_vars })
            D = xr.merge([K, p])
            D['period_start'] = int(period.start)
            D['period_end'] = int(period.stop)
        else:
            D = K.to_dataset(name='K_obs')
            D['p-value'] = p
            D['period_start'] = int(period.start)
            D['period_end'] = int(period.stop)
            
        return D
    
    fcst_ds = fcst.sel(time=max_period)
    obsv_ds = obsv.sel(time=max_period)

    if by_lead:
        return fcst_ds.groupby(fcst_ds.lead_time).map(
            _get_KS_statistic, obsv_ds=obsv_ds, by_lead=by_lead)
    else:
        return _get_KS_statistic(fcst_ds, obsv_ds, by_lead=by_lead)
    
    
def fidelity_KS_bivariate(fcst_var1, fcst_var2, obsv_var1, obsv_var2, max_period, by_lead, n_bootstraps=10_000):
    """ Perform a Kolmogorov-Smirnov test on bivariate data """
    def _get_KS_statistic(fcst_ds, obsv_ds, by_lead=False):
        if by_lead:
            # Applied per lead_time within groupby
            stack_dim = [d for d in fcst_ds.dims if 'stacked_' in d][0]
            fcst_ds = fcst_ds.assign_coords({stack_dim: fcst_ds[stack_dim].time})
            fcst_ds = fcst_ds.rename({stack_dim: 'time'})
            sample_dims = ['ensemble','time']
        else:
            sample_dims = ['ensemble','init_date','time']

        # Adjust period to range of available data
        min_overlap_year = max(
            fcst_ds.time.dt.year.min().item(),
            obsv_ds.time.dt.year.min().item())
        max_overlap_year = min(
            fcst_ds.time.dt.year.max().item(),
            obsv_ds.time.dt.year.max().item())
        period = slice(str(min_overlap_year), str(max_overlap_year))

        fcst_ds_samples = fcst_ds.sel(time=period).stack(
            sample=sample_dims).dropna('sample')
        obsv_ds_samples = obsv_ds.sel(time=period).rename({'time': 'sample'})

        D = xks.ks2d2s(obsv_ds_samples, fcst_ds_samples, 'sample')
        D_mc = n_random_resamples(fcst_ds_samples,
                                     samples={'sample': (len(obsv_ds_samples.sample), 1)}, 
                                     n_repeats=n_bootstraps,
                                     function=xks.ks2d2s,
                                     function_kwargs={'ds2': fcst_ds_samples, 'sample_dim': 'sample'},
                                     with_dask=True)

        D = D.to_dataset(name='K_obs')
        D['K'] = D_mc
        D['period_start'] = int(period.start)
        D['period_end'] = int(period.stop)
        return D

    assert fcst_var1.sizes == fcst_var2.sizes
    assert obsv_var1.sizes == obsv_var2.sizes

    if isinstance(fcst_var1, xr.DataArray):
        # Assume all are DataArrays
        assert fcst_var1.name == obsv_var1.name
        assert fcst_var2.name == obsv_var2.name
        
    elif isinstance(fcst_var1, xr.Dataset):
        # Assume all are Datasets
        fcst_var1_vars = list(fcst_var1.data_vars)
        fcst_var2_vars = list(fcst_var2.data_vars)
        obsv_var1_vars = list(obsv_var1.data_vars)
        obsv_var2_vars = list(obsv_var2.data_vars)
        assert len(fcst_var1_vars) == 1
        assert len(fcst_var2_vars) == 1
        assert fcst_var1_vars == obsv_var1_vars
        assert fcst_var2_vars == obsv_var2_vars
    else:
        raise InputError('Input arrays must be xarray DataArrays or Datasets')

    fcst_ds = xr.merge([fcst_var1.sel(time=max_period), fcst_var2.sel(time=max_period)])
    obsv_ds = xr.merge([obsv_var1.sel(time=max_period), obsv_var2.sel(time=max_period)])

    if by_lead:
        return fcst_ds.groupby(fcst_ds.lead_time).map(
            _get_KS_statistic, obsv_ds=obsv_ds, by_lead=by_lead)
    else:
        return _get_KS_statistic(fcst_ds, obsv_ds, by_lead=by_lead)
    

def likelihoods_of_exceedance(*variables, event=None, with_dask=False):
    """ Get empirical likelihoods of exceeding all combinations of the input variables.
        If event is provided, it should be a list of the same length as the number of
        variables
        For now, inputs should be one-dimensional
    """
    def _loe(reference):
        mask = variables[0] >= reference[0]
        for i in range(1,len(reference)):
            mask &= variables[i] >= reference[i]
        return 100 * mask.mean().values
    
    if event is None:
        references = zip(*variables)
        single_event = False
    else:
        if not isinstance(event, list):
            raise ValueError('event should be a list of the same length as the number of variables')
        assert len(variables) == len(event)
        references = [event]
        single_event = True

    if with_dask & (single_event == False):
        b = db.from_sequence(references, npartitions=100)
        likelihoods = np.array(b.map(_loe).compute())
    else:
        likelihoods = np.array([_loe(reference) for reference in references])
    
    # Package back up as xarray object
    dim = variables[0].dims[0]
    if event is None:
        coords = {v.name: ([dim], v.values) for v in variables}
        coords = {**coords, dim: ([dim], range(len(variables[0][dim])))}
    else:
        coords = {v.name: ([dim], [v.item()]) for v in event}
        coords = {**coords, dim: ([dim], [0])}
    return xr.DataArray(likelihoods, dims=[dim], coords=coords)

    
# =======================================================================
# Utilities
# =======================================================================

def get_region(ds, region):
    """ Return a region from a provided DataArray or Dataset
        
        Parameters
        ----------
        region_mask: xarray DataArray or list
            Boolean mask of the region to keep
    """
    return ds.where(region, drop=True)


def sum_min_samples(ds, dim, min_samples):
    """ Return sum only if there are more than min_samples along dim """
    s = ds.sum(dim, skipna=False)
    # Reference lead_time coord to final lead in sample
    if 'lead_time' in ds.coords:
        if dim in ds['lead_time'].dims:
            l = ds['lead_time'].max(dim, skipna=False)
            s = s.assign_coords({'lead_time': l if len(ds[dim]) >= min_samples else np.nan*l})
    return s if len(ds[dim]) >= min_samples else np.nan*s


def mean_min_samples(ds, dim, min_samples):
    """ Return mean only if there are more than min_samples along dim """
    m = ds.mean(dim, skipna=False)
    # Reference lead_time coord to final lead in sampl
    if 'lead_time' in ds.coords:
        if dim in ds['lead_time'].dims:
            l = ds['lead_time'].max(dim, skipna=False)
            m = m.assign_coords({'lead_time': l if len(ds[dim]) >= min_samples else np.nan*l})
    return m if len(ds[dim]) >= min_samples else np.nan*m


def resample_months_in_year(ds, months, method, time_dim='time', lead_time_dim='lead_time'):
    """ Resample monthly forecasts to a set of months for each year 
        This approach is hoaky but much faster/more memory efficient than using resample
    """
    # Clip to first instance of first month
    min_date = ds[time_dim].min()
    min_year = min_date.dt.year.values
    min_month = min_date.dt.month.values
    start_time = f'{min_year}-{months[0]:02d}' if min_month <= months[0] else f'{min_year+1}-{months[0]:02d}'
    ds = ds.copy().sel({time_dim:slice(start_time, None)})

    # Create mask of months to keep
    keep = ds[time_dim].dt.month == months[0]
    for month in months[1:]:
        keep = keep | (ds[time_dim].dt.month == month)
    ds = ds.where(keep, drop=True)
    
    # Use coarsen to do resampling
    return getattr(
        ds.coarsen(
            {time_dim: len(months)}, 
            boundary='trim',
            coord_func={time_dim: 'max',
                        lead_time_dim: 'max'}), 
        method)(skipna=False)


def calc_DEC_average(ds):
    """ Return the Dec average from daily data, excluding incomplete months """
    ds_mon = ds.resample(
        time="M", label='right').apply(
        mean_min_samples, dim='time', min_samples=31)
    return ds_mon.where(ds_mon.time.dt.month == 12, drop=True)
#     return _resample_months_in_year(ds, months=[12]).apply(
#         mean_min_samples, dim='time', min_samples=31)


def calc_ANN_accum(ds):
    """ Return the Jan-Dec accumulation from daily data, excluding incomplete months """
    return ds.resample(
        time="A-DEC", label='right').apply(
        sum_min_samples, dim='time', min_samples=365)


def truncate_latitudes(ds):
    for dim in ds.dims:
        if 'lat' in dim:
            ds = ds.assign_coords({dim: ds[dim].round(decimals=10)})
    return ds


def round_to_end_of_month(ds, dim='time'):
    from xarray.coding.cftime_offsets import MonthEnd
    return ds.assign_coords({dim: ds[dim].dt.floor('D') + MonthEnd()})


def interpolate_na_times(ds, coord='time'):
    """ Linearly interpolate any NaNs in time coordinate """
    def _date2num(time, units, calendar, has_year_zero):
        if time != time:
            return time
        else:
            return cftime.date2num(time, units, calendar, has_year_zero)
        
    _vdate2num = np.vectorize(_date2num)

    def _num2date(time, units, calendar, has_year_zero):
        return cftime.num2date(time, units, calendar, has_year_zero)
    
    _vnum2date = np.vectorize(_num2date)

    ds_cpy = ds.copy()

    units = 'days since 1900-01-01'
    # Assumes all calendar entries are the same
    calendar = ds_cpy[coord].values.flat[0].calendar
    has_year_zero = ds_cpy[coord].values.flat[0].has_year_zero
    
    ds_cpy.time.values = _vdate2num(ds_cpy.time.values, units, calendar, has_year_zero)
    ds_cpy = ds_cpy.assign_coords(
        {'time': ds_cpy.time.interpolate_na(dim='lead_time', 
                                            method="linear", 
                                            fill_value="extrapolate")})
    ds_cpy.time.values = _vnum2date(ds_cpy.time.values, units, calendar, has_year_zero)
    
    # Check that all values that changed were nans
    assert ds.time.where(ds_cpy.time != ds.time).isnull().all()
    
    return ds_cpy


def estimate_cell_areas(ds, lon_dim='lon', lat_dim='lat'):
    """
    Calculate the area of each grid cell
    
    Stolen/adapted from: https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7
    """
    
    def _earth_radius(lat):
        """ Calculate radius of Earth assuming oblate spheroid defined by WGS84
        """
        from numpy import deg2rad, sin, cos

        # define oblate spheroid from WGS84
        a = 6378137
        b = 6356752.3142
        e2 = 1 - (b**2/a**2)

        # convert from geodecic to geocentric
        # see equation 3-110 in WGS84
        lat = xr.ufuncs.deg2rad(lat)
        lat_gc = xr.ufuncs.arctan( (1-e2)*xr.ufuncs.tan(lat) )

        # radius equation
        # see equation 3-107 in WGS84
        return ((a * (1 - e2)**0.5) 
             / (1 - (e2 * xr.ufuncs.cos(lat_gc)**2))**0.5)

    R = _earth_radius(ds[lat_dim])

    dlat = xr.ufuncs.deg2rad(ds[lat_dim].diff(lat_dim))
    dlon = xr.ufuncs.deg2rad(ds[lon_dim].diff(lon_dim))

    dy = dlat * R
    dx = dlon * R * xr.ufuncs.cos(xr.ufuncs.deg2rad(ds[lat_dim]))

    return dy * dx


def lat_lon_average(ds, box, area, lat_dim='lat', lon_dim='lon'):
    """ Get average over lat lon region """
    def _get_lat_lon_region(ds, box, lat_dim, lon_dim):
        region = (
            (ds[lat_dim] >= box[0]) & (ds[lat_dim] <= box[1])
        ) & (
            (ds[lon_dim] >= box[2]) & (ds[lon_dim] <= box[3]))
        return ds.where(region)
    
    ds = ds.assign_coords({lon_dim: (ds[lon_dim] + 360)  % 360})

    if area is None:
        area = estimate_cell_areas(ds, lon_dim, lat_dim)
        
    if (lat_dim in ds.dims) and (lon_dim in ds.dims):
        # lat and lon are dims so use isel to get regions
        lon_inds = np.where(
            np.logical_and(ds[lon_dim].values>=box[2], 
                           ds[lon_dim].values<=box[3]))[0]
        lat_inds = np.where(
            np.logical_and(ds[lat_dim].values>=box[0], 
                           ds[lat_dim].values<=box[1]))[0]
        return ds.isel(
            {lon_dim: lon_inds, lat_dim: lat_inds}).weighted(
            area).mean(
            dim=[lat_dim, lon_dim])
    elif (lat_dim in ds.dims) and (lon_dim not in ds.dims):
        return _get_lat_lon_region(
            ds, box, lat_dim, lon_dim).weighted(
            area).mean(
            dim=set([lat_dim, *ds[lon_dim].dims]))
    elif (lat_dim not in ds.dims) and (lon_dim in ds.dims):
        return _get_lat_lon_region(
            ds, box, lat_dim, lon_dim).weighted(
            area).mean(
            dim=set([*ds[lat_dim].dims, lon_dim]))
    else:
        return _get_lat_lon_region(
            ds, box, lat_dim, lon_dim).weighted(
            area).mean(
            dim=set([*ds[lat_dim].dims, *ds[lon_dim].dims]))
