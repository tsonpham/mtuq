#!/usr/bin/env python

import os
import numpy as np

from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens2, plot_beachball, plot_misfit_dc
from mtuq.grid import DoubleCoupleGridRegular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid



if __name__=='__main__':
    #
    # Carries out grid search over 64,000 double-couple moment tensors
    #
    # USAGE
    #   python SerialGridSearch.DoubleCouple.py
    #
    # A typical runtime is about 60 seconds. For faster results try 
    # GridSearch.DoubleCouple.py, which runs the same inversion in parallel
    #


    #
    # We will investigate the source process of an Mw~4 earthquake using data
    # from a regional seismic array
    #

    path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
    event_id=     '20090407201255351'
    model=        'ak135'


    #
    # Body and surface wave measurements will be made separately
    #

    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.1,
        freq_max= 0.333,
        pick_type='taup',
        taup_model=model,
        window_type='body_wave',
        window_length=15.,
        capuaf_file=path_weights,
        )

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='taup',
        taup_model=model,
        window_type='surface_wave',
        window_length=150.,
        capuaf_file=path_weights,
        )


    #
    # For our objective function, we will use a sum of body and surface wave
    # contributions
    #

    misfit_bw = Misfit(
        norm='L2',
        time_shift_min=-2.,
        time_shift_max=+2.,
        time_shift_groups=['ZR'],
        )

    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-10.,
        time_shift_max=+10.,
        time_shift_groups=['ZR','T'],
        )


    #
    # User-supplied weights control how much each station contributes to the
    # objective function
    #

    station_id_list = parse_station_codes(path_weights)


    #
    # Next, we specify the moment tensor grid and source-time function
    #

    grid = DoubleCoupleGridRegular(
        npts_per_axis=40,
        magnitudes=[4.5])

    wavelet = Trapezoid(
        magnitude=4.5)


    #
    # Origin time and location will be fixed. For an example in which they 
    # vary, see examples/GridSearch.DoubleCouple+Magnitude+Depth.py
    #
    # See also Dataset.get_origins(), which attempts to create Origin objects
    # from waveform metadata
    #

    origin = Origin({
        'time': '2009-04-07T20:12:55.000000Z',
        'latitude': 61.454200744628906,
        'longitude': -149.7427978515625,
        'depth_in_m': 33033.599853515625,
        })


    #
    # The main I/O work starts now
    #

    print('Reading data...\n')
    data = read(path_data, format='sac',
        event_id=event_id,
        station_id_list=station_id_list,
        tags=['units:cm', 'type:velocity']) 


    data.sort_by_distance()
    stations = data.get_stations()


    print('Processing data...\n')
    data_bw = data.map(process_bw)
    data_sw = data.map(process_sw)


    print('Reading Greens functions...\n')
    greens = download_greens_tensors(stations, origin, model)


    print('Processing Greens functions...\n')
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)


    #
    # The main computational work starts now
    #

    print('Evaluating body wave misfit...\n')
    results_bw = grid_search(data_bw, greens_bw, misfit_bw, origin, grid)

    print('Evaluating surface wave misfit...\n')
    results_sw = grid_search(data_sw, greens_sw, misfit_sw, origin, grid)


    #
    # Generate figures and save results
    #

    results = results_bw + results_sw

    # source corresponding to minimum misfit
    idx = results.idxmin('source')
    best_source = grid.get(idx)
    lune_dict = grid.get_dict(idx)
    mt_dict = grid.get(idx).as_dict()

    merged_dict = merge_dicts(lune_dict, mt_dict, origin)


    # only generate components present in the data
    components_bw = data_bw.get_components()
    components_sw = data_sw.get_components()

    # synthetics corresponding to minimum misfit
    synthetics_bw = greens_bw.get_synthetics(
        best_source, components_bw, mode='map')

    synthetics_sw = greens_sw.get_synthetics(
        best_source, components_sw, mode='map')


    # time shifts and other attributes corresponding to minimum misfit
    list_bw = misfit_bw.collect_attributes(
        data_bw, greens_bw, best_source)

    list_sw = misfit_sw.collect_attributes(
        data_sw, greens_sw, best_source)

    dict_bw = {station.id: list_bw[_i] 
        for _i,station in enumerate(stations)}

    dict_sw = {station.id: list_sw[_i] 
        for _i,station in enumerate(stations)}


    print('Generating figures...\n')

    plot_data_greens2(event_id+'DC_waveforms.png',
        data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
        misfit_bw, misfit_sw, stations, origin, best_source, lune_dict)

    plot_beachball(event_id+'DC_beachball.png', best_source)

    plot_misfit_dc(event_id+'DC_misfit.png', results)


    print('Saving results...\n')

    # save best-fitting source
    save_json(event_id+'DC_solution.json', merged_dict)


    # save time shifts and other attributes
    os.makedirs(event_id+'DC_attrs', exist_ok=True)

    save_json(event_id+'DC_attrs/bw.json', dict_bw)
    save_json(event_id+'DC_attrs/sw.json', dict_sw)


    # save processed waveforms as binary files
    os.makedirs(event_id+'DC_waveforms', exist_ok=True)

    data_bw.write(event_id+'DC_waveforms/dat_bw.p')
    data_sw.write(event_id+'DC_waveforms/dat_sw.p')

    synthetics_bw.write(event_id+'DC_waveforms/syn_bw.p')
    synthetics_sw.write(event_id+'DC_waveforms/syn_sw.p')

    results.save(event_id+'DC_misfit.nc')


    print('\nFinished\n')

