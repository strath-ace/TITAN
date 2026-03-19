from pathlib import Path

template = """[Options]
Num_iters = 100
Load_mesh = False
Load_state = False
Fidelity = Low
Output_folder = ControlTests/Results/Braking/{deg}deg
FENICS = False
Output_freq = 1
Postprocess_in_loop = WIND

[Mesh]
Recursion_limit = 5000

[Model]
Planet = Earth

[Trajectory]
Altitude = 50000
Velocity = 5000
Flight_path_angle = 0
Heading_angle = 0.0
Latitude = 0
Longitude = 0

[Freestream]
Model = NRLMSISE00
Method = Standard

[Time]
Time_step = 0.1
Propagator = euler

[Assembly]
Path = ControlTests/Geometry/
Connectivity = [[1, 6, 0],
                [1, 7, 0],
                [1, 8, 0],
                [1, 9, 0]]

[Objects]
Cube = [NAME = cubesatCube.stl, MATERIAL = Unittest, TYPE = Primitive, FENICS_ID = -1]
Hinge_L = [NAME = cubesatHinge_L.stl, MATERIAL = Unittest, TYPE = Joint, FENICS_ID = -1]
Hinge_R = [NAME = cubesatHinge_R.stl, MATERIAL = Unittest, TYPE = Joint, FENICS_ID = -1]
Hinge_T = [NAME = cubesatHinge_T.stl, MATERIAL = Unittest, TYPE = Joint, FENICS_ID = -1]
Hinge_B = [NAME = cubesatHinge_B.stl, MATERIAL = Unittest, TYPE = Joint, FENICS_ID = -1]
Flap_L = [NAME = cubesatFlap_L.stl, MATERIAL = Unittest, TYPE = ControlSurface, FENICS_ID = -1, DEFLECTION = -0, AXIS = (0,0,1), ORIGIN = (0, -.045, 0)]
Flap_R = [NAME = cubesatFlap_R.stl, MATERIAL = Unittest, TYPE = ControlSurface, FENICS_ID = -1, DEFLECTION = -0, AXIS = (0,0,-1), ORIGIN = (0,.045,0)]
Flap_T = [NAME = cubesatFlap_T.stl, MATERIAL = Unittest, TYPE = ControlSurface, FENICS_ID = -1, DEFLECTION = -0, AXIS = (0,-1,0), ORIGIN = (0,0,-.045)]
Flap_B = [NAME = cubesatFlap_B.stl, MATERIAL = Unittest, TYPE = ControlSurface, FENICS_ID = -1, DEFLECTION = -0, AXIS = (0,1,0), ORIGIN = (0, 0, .045)]

[Jets]
Roll_Jet_1_pos = [PARENT = cubesatCube, POS = (-.25,0.05,0), DIR = (0,0,1), TMAX = 10, ISP = 70, GROUP = roll_pos]
Roll_Jet_1_neg = [PARENT = cubesatCube, POS = (-.25,0.05,0), DIR = (0,0,-1), TMAX = 10, ISP = 70, GROUP = roll_neg]
Roll_Jet_2_pos = [PARENT = cubesatCube, POS = (-.25,-0.05,0), DIR = (0,0,-1), TMAX = 10, ISP = 70, GROUP = roll_pos]
Roll_Jet_2_neg = [PARENT = cubesatCube, POS = (-.25,-0.05,0), DIR = (0,0,1), TMAX = 10, ISP = 70, GROUP = roll_neg]

[Control]
Command_file = ControlTests/JetTests/showcase_commands.csv
Mode = time

[PropellantTanks]
MainTank = [PARENT = cubesatCube, TYPE = N2, CAPACITY = 10, INITIAL = 10, RESIDUAL = 0.1, POS = (-0.15, 0, 0), RADIUS = 1.0, DRY_MASS = 0.5]
"""

out_dir = Path(".")
for deg in range(90, 35, -5):
    fname = out_dir / f"{deg}deg.txt"
    fname.write_text(template.format(deg=deg))

print("done!")