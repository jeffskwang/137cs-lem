#importing libaries
import numpy as np
import time
import os
from osgeo import gdal, osr, gdalnumeric

sitename = 'test'

#make output folder
out_dir = 'output/'+sitename
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#timer
start_time = time.time()

#input data
input_dem_f = 'data/dem.tif'

###physical parameters###
D = 0.2#[m^2/yr] diffusion coefficient
U = 0.0 #[-] dimensionless uplift rate
La = 0.25 #[m] mixing layer
dt = 1. # [-] model timestep

#137Cs parameters
#simulation start
T_start = 1954 #[yr]
T_end = 2002 #[yr]
dt_plot = 12.# [-] plot timestep
Cs_lambda = np.log(2.) / 30.

#137Cs input rate
depo_data = np.loadtxt('data/deposition_137Cs.csv',skiprows=1,delimiter=',')
input_time = depo_data[:,0] #[yr] start of cs input interval
input_rate = depo_data[:,3]# [becquerels/m^2/yr]

T = T_end - T_start# [yr] simulation time

#load landlab components
from landlab import RasterModelGrid

def load_data_tif(dem_fname):
    """
    Load GeoTIFF with gdal and import into landlab
    """
    t = gdal.Open(dem_fname)
    gt = t.GetGeoTransform()
    cs = t.GetProjection()
    cs_sr = osr.SpatialReference()
    cs_sr.ImportFromWkt(cs)
    dem = gdalnumeric.LoadFile(dem_fname).astype(float)
    nr_of_x_cells =dem.shape[1]
    nr_of_y_cells =dem.shape[0]
    
    if gt[5] < 0:
        dem = np.flipud(dem)
        
    dem_padded = np.zeros((dem.shape[0] + 2,dem.shape[1] + 2))
    dem_padded[:,:]=-9999
    dem_padded[1:-1,1:-1] = dem
        
    mg = RasterModelGrid((nr_of_y_cells+2,nr_of_x_cells+2), xy_spacing=(abs(gt[1]), abs(gt[1])))
    eta = mg.add_field('topographic__elevation', dem_padded, at = 'node')
    eta[eta==np.min(eta)]=-9999 #minimum value is no data, either -9999 or -3.40282346639e+38

    return mg, eta,gt, cs, nr_of_x_cells, nr_of_y_cells

def write_data_tif(dem_out_fname,eta_in,gt,cs,nr_of_x_cells,nr_of_y_cells):   
    eta = np.flipud(eta_in.reshape(nr_of_y_cells+2,nr_of_x_cells+2))[1:-1,1:-1]
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(dem_out_fname,nr_of_x_cells,nr_of_y_cells, 1,gdal.GDT_Float32)
    dataset.GetRasterBand(1).WriteArray(eta)
    dataset.GetRasterBand(1).SetNoDataValue(-9999)

    dataset.SetGeoTransform(gt)
    dataset.SetProjection(cs)
    dataset.FlushCache()
    dataset=None 
 

#load initial condition
grid, eta,gt, cs, nr_of_x_cells, nr_of_y_cells = load_data_tif(input_dem_f)
grid.set_nodata_nodes_to_closed(eta, -9999)
grid.set_fixed_value_boundaries_at_grid_edges(True,True,True,True)
nrows = grid.number_of_node_rows
ncols = grid.number_of_node_columns
#define arrays
SOC_La = grid.add_zeros('node','SOC_La')
SOC_transfer = grid.add_zeros('node','SOC_transfer')
dqda = grid.add_zeros('node','dqda')
dz = 0.01
Z = 1.0
nz = int(2.0 * Z / dz)
SOC_z = np.zeros((nrows,ncols,nz))
dz_ini = np.zeros((nrows,ncols,nz))

#grid size and number of cells
dx = grid.dx
dy = grid.dy
nt = int(T / dt)
nt_plot = int(dt_plot/dt)

#########################

#####FUNCTIONS#####

def IC(eta, dz_ini, SOC_z, SOC_La):
    for i in range(0,nrows):
        for j in range(0,ncols):
            dz_ini[i,j,:] = np.linspace(-Z + dz / 2.0, Z - dz / 2.0, nz)
            SOC_z[i,j,0:int(nz/2)] = 0.0#np.linspace(0.0,5.0,int(nz/2))      
            avg_count = 0
            for k in range(int(nz/2 - La / dz),int(nz/2)):
                avg_count += 1
                SOC_La.reshape(nrows,ncols)[i,j] += SOC_z[i,j,k]
            SOC_La.reshape(nrows,ncols)[i,j] /= float(avg_count)
     
    return dz_ini, SOC_z, SOC_La

def boundary_conditions(eta):
    noflow = grid.map_value_at_min_node_to_link(eta,eta)
    noflow[noflow!=-9999] = 1.0
    noflow[noflow==-9999] = 0.0
    return noflow

def soil_and_SOC_transport(eta,SOC_La,noflow):
    dzdx = grid.calc_grad_at_link(eta)
    SOC_La_uphill = grid.map_value_at_max_node_to_link(eta,SOC_La)    
    q = - D * dzdx * noflow
    qc = q * SOC_La_uphill * noflow
    dqda = grid.calc_flux_div_at_node(q)
    dqcda = grid.calc_flux_div_at_node(qc)
    return dqda,dqcda

def find_SOC_cell(interface_z,z_coor,SOC):
    index_locat = (np.abs(z_coor - interface_z)).argmin()
    SOC_interface = SOC[index_locat]
    
    return SOC_interface  

def find_top_cell_active_layer(interface_z,z_coor,SOC):
    top_cell_active_layer = (np.abs(z_coor - interface_z + dz /2.)).argmin()
    
    return top_cell_active_layer        

def find_bottom_cell_active_layer(interface_z,z_coor,SOC):
    bottom_cell_active_layer = (np.abs(z_coor - interface_z)).argmin()
    
    return bottom_cell_active_layer

def SOC_transfer_function(eta_old,eta_ini,dzdt,SOC_La,SOC_transfer,SOC_z):
    interface = eta_old.reshape(nrows,ncols) - eta_ini.reshape(nrows,ncols) - La
    for i in range(0,nrows):
        for j in range(0,ncols):
            if dzdt.reshape(nrows,ncols)[i,j] < 0.0:
                SOC_transfer.reshape(nrows,ncols)[i,j] = find_SOC_cell(interface.reshape(nrows,ncols)[i,j],dz_ini[i,j,:],SOC_z[i,j,:])
##            elif dzdt.reshape(nrows,ncols)[i,j] > 0.0: TEMPORARY FIX
##                SOC_transfer.reshape(nrows,ncols)[i,j] = SOC_La.reshape(nrows,ncols)[i,j]
    return SOC_transfer

def SOC_profile_update(eta,eta_ini,dzdt,SOC_La,SOC_z):
    interface = eta - eta_ini - La
    for i in range(0,nrows):
        for j in range(0,ncols):
            top_cell_active_layer =  find_top_cell_active_layer(interface.reshape(nrows,ncols)[i,j]+La,dz_ini[i,j,:],SOC_z[i,j,:])
            bottom_cell_active_layer =  find_bottom_cell_active_layer(interface.reshape(nrows,ncols)[i,j],dz_ini[i,j,:],SOC_z[i,j,:])
            SOC_z[i,j,top_cell_active_layer+1:] = 0.0
            SOC_z[i,j,bottom_cell_active_layer+1:top_cell_active_layer+1]=SOC_La.reshape(nrows,ncols)[i,j]

            if dzdt.reshape(nrows,ncols)[i,j] > 0.0:
                dz_interface_old = (interface.reshape(nrows,ncols)[i,j] - dt * dzdt.reshape(nrows,ncols)[i,j]) - dz_ini[i,j,bottom_cell_active_layer]  + dz / 2.0
                dz_interface_new = interface.reshape(nrows,ncols)[i,j] - dz_ini[i,j,bottom_cell_active_layer]  + dz / 2.0
                SOC_z[i,j,bottom_cell_active_layer] = (SOC_z[i,j,bottom_cell_active_layer] * dz_interface_old + dt * dzdt.reshape(nrows,ncols)[i,j] *  SOC_La.reshape(nrows,ncols)[i,j]) / (dz_interface_new)
                
    return SOC_z

def SOC_input(t):
    time_in_years = t * dt
    calendar_year = time_in_years + T_start
    if calendar_year >= input_time[-1]:
        current_input_rate = input_rate[-1]
    else:
        for i in range(0,len(input_time)-1):
            if calendar_year >= input_time[i] and calendar_year< input_time[i+1]:
                current_input_rate = input_rate[i]
    return current_input_rate
        
            
##### LOOP START #####
noflow = boundary_conditions(eta)
dz_ini, SOC_z, SOC_La = IC(eta, dz_ini, SOC_z, SOC_La)
SOC_La[grid.boundary_nodes] = -9999
eta_ini = eta.copy()
print ('loading',str(round((time.time() -start_time)/60.,1))+' mins')
for t in range(0,nt + 1):
    t_step_start = time.time()
    if t%nt_plot == 0:
        write_data_tif('output/'+sitename+'/'+sitename+'_eta_'+ '%06d' % t +'yrs.tif',eta,gt,cs,nr_of_x_cells,nr_of_y_cells)
        write_data_tif('output/'+sitename+'/'+sitename+'_soc_'+ '%06d' % t +'yrs.tif',SOC_La,gt,cs,nr_of_x_cells,nr_of_y_cells)
    eta_old = eta.copy()
    dqda,dqcda = soil_and_SOC_transport(eta,SOC_La,noflow)
    eta[grid.core_nodes] += dt *(U - dqda[grid.core_nodes])
    dzdt = (eta - eta_old)/dt
    SOC_transfer = SOC_transfer_function(eta_old,eta_ini,dzdt,SOC_La,SOC_transfer,SOC_z)
    SOC_La[grid.core_nodes]  += dt/La * (SOC_transfer[grid.core_nodes] * dqda[grid.core_nodes]  - dqcda[grid.core_nodes]) + dt * (SOC_input(t) / La  - Cs_lambda * SOC_La[grid.core_nodes])
    SOC_z  = SOC_profile_update(eta,eta_ini,dzdt,SOC_La,SOC_z)        
    t_step_end = time.time()    
    print (str(t) +' yrs',str(round((t_step_end -t_step_start)/60.,1))+' mins')

stop_time = time.time()
print (str(round((stop_time -start_time )/60.,1))+' mins')
