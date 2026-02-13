from matplotlib import pyplot as plt
import numpy as np

def initialise_figs(titan, options, geodetic_pos=True):
    plt.ion()
    # for some reason np.pi breaks unless np is re-imported here, frankly no clue
    import numpy as np
    plt.style.use('dark_background')
    fig, ax  = plt.subplots()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')
    aoa_plot, = ax.plot([titan.time],[titan.assembly[0].aoa], label="Angle of Attack",color='g',linestyle=':',marker = 'x')
    ss_plot, = ax.plot([titan.time],[titan.assembly[0].slip], label='Sideslip',color='c',linestyle=':',marker = 'o')
    aoas = [titan.assembly[0].aoa]
    sss = [titan.assembly[0].slip]
    times = [0.0]
    positions = titan.assembly[0].position
    sim_name = options.output_folder.split('/')[-1]
    ax.set_title("{} | Attitude".format(sim_name))
    ax.set_xlabel("Time")
    ax.set_ylabel("Angle (Deg)")
    if geodetic_pos:
        pi = np.pi
        position_plot = ax2.plot([titan.assembly[0].trajectory.longitude*180/pi],[titan.assembly[0].trajectory.latitude*180/pi],[titan.assembly[0].trajectory.altitude/1000],
                                color='w',linestyle='None',marker = 'o')[0]
        ax2.set_xlabel("Lon (Deg)")
        ax2.set_ylabel("Lat (Deg)")
        ax2.set_zlabel("Alt (km)")
    else:
        position_plot = ax2.plot([titan.assembly[0].position[0]],[titan.assembly[0].position[1]],[titan.assembly[0].position[2]],
                                color='w',linestyle='None',marker = 'o')[0]
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_zlabel("Z (m)")
        import numpy as np
        theta, phi = np.linspace(0, 2 * np.pi, 60), np.linspace(0, np.pi, 50)
        THETA, PHI = np.meshgrid(theta, phi)
        R = 6378000
        X = R * np.sin(PHI) * np.cos(THETA)
        Y = R * np.sin(PHI) * np.sin(THETA)
        Z = R * np.cos(PHI)
        plot = ax2.plot_wireframe(X, Y, Z, rstride=1, linewidth=1.0, color='green', alpha=0.6)
    ax2.set_title("{} | Spacial Position".format(sim_name))

    ax2.xaxis.set_pane_color([0,0,0])
    ax2.yaxis.set_pane_color([0,0,0])
    ax2.zaxis.set_pane_color([0,0,0])
    ax.legend()
    ax.grid(True)


    
    plot_parameters = {'fig1':fig,'fig2':fig2,'times':times,'aoa':aoas,'ss':sss,'pos':positions,'aoaplot':aoa_plot,'ssplot':ss_plot,'posplot':position_plot,'ax1':ax,'ax2':ax2}
    return plot_parameters

def update_plot(assembly,plot_parameters, time, geodetic_pos=True):

    fig = plot_parameters['fig1']
    fig2 = plot_parameters['fig2']
    ax = plot_parameters['ax1']
    ax2 = plot_parameters['ax2']
    times = plot_parameters['times']
    aoas = plot_parameters['aoa']
    sss = plot_parameters['ss']
    positions = plot_parameters['pos']
    aoa_plot = plot_parameters['aoaplot']
    ss_plot = plot_parameters['ssplot']
    position_plot = plot_parameters['posplot']


    aoas.append(assembly.aoa*(360/(2*3.14159)))
    sss.append(assembly.slip*(360/(2*3.14159)))
    if geodetic_pos: positions = np.vstack(([assembly.trajectory.longitude*180/np.pi,assembly.trajectory.latitude*180/np.pi,assembly.trajectory.altitude/1000],positions))
    else: positions = np.vstack((assembly.position,positions))

    times.append(time)
    aoa_plot.set_data(times,aoas)
    ss_plot.set_data(times,sss)
    position_plot.set_data_3d(positions[:,0],positions[:,1],positions[:,2])
    ax.relim()
    ax.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig2.canvas.draw()
    fig2.canvas.flush_events()

    plot_parameters = {'fig1':fig,'fig2':fig2,'times':times,'aoa':aoas,'ss':sss,'pos':positions,'aoaplot':aoa_plot,'ssplot':ss_plot,'posplot':position_plot,'ax1':ax,'ax2':ax2}
    return plot_parameters

