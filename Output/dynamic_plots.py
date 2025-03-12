from matplotlib import pyplot as plt
import numpy as np

def initialise_figs(titan):
    plt.ion()
    plt.style.use('dark_background')
    fig, ax  = plt.subplots()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')
    aoa_plot, = ax.plot([titan.time],[titan.assembly[0].aoa], label="Angle of Attack",color='g',linestyle=':',marker = 'x')
    ss_plot, = ax.plot([titan.time],[titan.assembly[0].slip], label='Sideslip',color='c',linestyle=':',marker = 'o')
    position_plot = ax2.plot([titan.assembly[0].position[0]],[titan.assembly[0].position[1]],[titan.assembly[0].position[2]],
                                color='w',linestyle='None',marker = 'o')[0]
    aoas = [titan.assembly[0].aoa]
    sss = [titan.assembly[0].slip]
    times = [0.0]
    positions = titan.assembly[0].position
    ax.set_title("Live plot updates!")
    ax.set_xlabel("Time")
    ax.set_ylabel("Angle (Deg)")
    ax2.set_title("Spacial Position")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.xaxis.set_pane_color([0,0,0])
    ax2.yaxis.set_pane_color([0,0,0])
    ax2.zaxis.set_pane_color([0,0,0])
    ax.legend()
    ax.grid(True)
    import numpy as np
    theta, phi = np.linspace(0, 2 * np.pi, 60), np.linspace(0, np.pi, 50)
    THETA, PHI = np.meshgrid(theta, phi)
    R = 6378000
    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)

    plot = ax2.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=1.0, shade=False, color='green', alpha=0.6)
    plot_parameters = {'fig1':fig,'fig2':fig2,'times':times,'aoa':aoas,'ss':sss,'pos':positions,'aoaplot':aoa_plot,'ssplot':ss_plot,'posplot':position_plot,'ax1':ax,'ax2':ax2}
    return plot_parameters

def update_plot(assembly,plot_parameters, time):

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
    positions = np.vstack((assembly.position,positions))
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

