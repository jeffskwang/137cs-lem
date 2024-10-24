from osgeo import gdal, osr
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from natsort import natsorted
import glob
from dbfread import DBF
import pandas as pd

output_folder = 'output/'
runname = 'test'
max_soc = 5000.
max_eta = 0.5

fig_width = 12.0
fig_height = 10.0

#matplot settings
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'axes.linewidth': 1})
matplotlib.rcParams.update({'xtick.major.width':1})
matplotlib.rcParams.update({'ytick.major.width': 1})

#plot eta
def plot_eta_diff(runname):
    fig = plt.figure(figsize=(fig_width,fig_height))
    first_file = 1
    for file in os.listdir(output_folder + runname):
        if file.startswith(runname+'_eta') and file.endswith('.tif'):
            ds = gdal.Open(output_folder + runname+'//'+file)
            data = ds.ReadAsArray()
            data[data==-9999] = np.nan
            gt = ds.GetGeoTransform()
            extent = (gt[0], gt[0] + ds.RasterXSize * gt[1],
                  gt[3] + ds.RasterYSize * gt[5], gt[3])

            if first_file == 1:
                first_file = 0
                data_IC = data

            plt.imshow(data - data_IC,extent=extent,vmin=-max_eta,vmax=max_eta,cmap='seismic_r')
            plt.colorbar(label=r'$\eta$ [$m$]')
            plt.xlabel('Easting [m]')
            plt.ylabel('Northing [m]')
            plt.tight_layout()
            plt.savefig(output_folder + runname+'//plot_'+file[:-4]+'.png',dpi=300)
            plt.clf()
            
#plot soc
def plot_soc(runname):
    fig = plt.figure(figsize=(fig_width,fig_height))
    for file in os.listdir(output_folder + runname):
        if file.startswith(runname+'_soc') and file.endswith('.tif'):
            ds = gdal.Open(output_folder + runname+'//'+file)
            data = ds.ReadAsArray()
            data[data==-9999] = np.nan
            gt = ds.GetGeoTransform()
            extent = (gt[0], gt[0] + ds.RasterXSize * gt[1],
                  gt[3] + ds.RasterYSize * gt[5], gt[3])
            plt.imshow(data * 0.25,extent=extent,vmin=0.0,vmax=max_soc)
            plt.colorbar(label=r'$SOC$ [$g/cm^3$]')
            plt.xlabel('Easting [m]')
            plt.ylabel('Northing [m]')
            plt.tight_layout()
            plt.savefig(output_folder + runname+'//plot_'+file[:-4]+'.png',dpi=300)
            plt.clf()

plot_eta_diff(runname)
plot_soc(runname)



shpfl = glob.glob('data/XYdata_137Cs.dbf')[0]
dbf = DBF(shpfl)
shp_df = pd.DataFrame(iter(dbf))
cs_conc = shp_df.cesium
east_pos = shp_df.Easting
north_pos = shp_df.Northing
delta_eta = shp_df.redist / shp_df.bulk_dens / 1000. * 48.#meters

def bilinear_interp(x,y,data):
    x_lower = int(x)
    y_lower = int(y)
    x_upper = x_lower + 1
    y_upper = y_lower + 1
    
    step_1 = (float(data[x_upper,y_lower]) - float(data[x_lower,y_lower])) / 1.0 * (x - float(x_lower)) + data[x_lower,y_lower]
    step_2 = (float(data[x_upper,y_upper]) - float(data[x_lower,y_upper])) / 1.0 * (x - float(x_lower)) + data[x_lower,y_upper]

    step_3 =  (step_2 - step_1) / 1.0 * (y - float(y_lower)) + step_1

    return step_3

def plot_fit(east_pos,north_pos,cs_conc,delta_eta,runname):
    
    ds = gdal.Open(output_folder + runname+'//'+runname+'_soc_000048yrs.tif')
    data = ds.ReadAsArray()
    data[data==-9999] = np.nan
    gt = ds.GetGeoTransform()
    ul_easting = gt[0]
    ul_northing = gt[3]
    cs_conc_model = np.zeros_like(cs_conc)
    delta_eta_model = np.zeros_like(delta_eta)

    ds1 = gdal.Open(output_folder + runname+'//'+runname+'_eta_000000yrs.tif')
    data1 = ds1.ReadAsArray()
    data1[data1==-9999] = np.nan
    
    ds2 = gdal.Open(output_folder + runname+'//'+runname+'_eta_000048yrs.tif')
    data2 = ds2.ReadAsArray()
    data2[data2==-9999] = np.nan

    data_eta = data2 - data1
    
    for i in range(0,len(east_pos)):
        x_loc = float((east_pos[i]-ul_easting)/gt[1])-0.5
        y_loc = float((north_pos[i]-ul_northing)/gt[5])-0.5
        cs_conc_model[i] = bilinear_interp(y_loc,x_loc,data) * 0.25
        delta_eta_model[i] = bilinear_interp(y_loc,x_loc,data_eta)

    fig = plt.figure(1,figsize=(6,6))
    plt.scatter(cs_conc_model,cs_conc)
    plt.plot([500,4000],[500,4000])
    plt.xlabel(r'model $^{137}Cs$ [$Bq/m^2$]')
    plt.ylabel(r'measured $^{137}Cs$ [$Bq/m^2$]')
    plt.tight_layout()
    plt.savefig(output_folder + runname+'//data_137cs_fit.png',dpi=300)

    fig = plt.figure(2,figsize=(6,6))
    plt.scatter(delta_eta_model,delta_eta)
    plt.plot([np.min(delta_eta_model[~np.isnan(delta_eta_model)]),np.max(delta_eta_model[~np.isnan(delta_eta_model)])],[np.min(delta_eta_model[~np.isnan(delta_eta_model)]),np.max(delta_eta_model[~np.isnan(delta_eta_model)])])
    plt.xlabel(r'model $\Delta\eta$ [m]')
    plt.ylabel(r'measured $\Delta\eta$ [m]')
    plt.tight_layout()
    plt.savefig(output_folder + runname+'//data_eta_fit.png',dpi=300)
    
    ##    extent = (gt[0], gt[0] + ds.RasterXSize * gt[1],
    ##          gt[3] + ds.RasterYSize * gt[5], gt[3])
    ##    plt.imshow(data * 0.25,extent=extent,vmin=0.0,vmax=max_soc)
    ##    plt.scatter(east_pos,north_pos,color='r',s=0.5)
    ##    plt.colorbar(label=r'$SOC$ [$g/cm^3$]')
    ##    plt.xlabel('Easting [m]')
    ##    plt.ylabel('Northing [m]')

plot_fit(east_pos,north_pos,cs_conc,delta_eta,runname)