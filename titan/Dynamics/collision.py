"""collision module."""
import trimesh
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as Rot
import pyquaternion

class Collision():
	"""Collision."""
	def __init__(self):
		"""__init__."""
		self.collision_mesh = None
		self.original_mesh = None
		self.collision_handler = None
		self.original_handler =  None

def generate_collision_handler(titan, options):
	'''
Creates trimesh collision handlers (managers) for assemblies in the list
	:param titan: Object of class AssemblyList
	:type titan: object
	:param options: Object of class Options
	:type options: object
'''
	for assembly in titan.assembly:
		assembly.collision.collision_handler = trimesh.collision.CollisionManager()
		assembly.collision.original_handler = trimesh.collision.CollisionManager()
		assembly.collision.collision_handler.add_object("Collision_"+str(assembly.id), np.sum(assembly.collision.collision_mesh))
		assembly.collision.original_handler.add_object("Original_"+str(assembly.id), np.sum(assembly.collision.original_mesh))

def delete_collision_handler(titan, options):
	'''
Deletes trimesh collision handlers (managers) from assemblies in the list
	:param titan: Object of class AssemblyList
	:type titan: object
	:param options: Object of class Options
	:type options: object
'''
	for assembly in titan.assembly:
		assembly.collision.collision_handler = None
		assembly.collision.original_handler = None

def update_collision_mesh(titan, options):
	'''
Transforms assembly collision meshes into local ECEF frame of largest mass assembly
	:param titan: Object of class AssemblyList
	:type titan: object
	:param options: Object of class Options
	:type options: object
'''
	mass = []

	for assembly in titan.assembly:
		mass.append(assembly.mass)

	index_mass = np.argmax(mass)
	Translate_Large_Mass = trimesh.transformations.translation_matrix(-titan.assembly[index_mass].position)

	for assembly in titan.assembly:
		quaternion = np.append([assembly.quaternion[3]], assembly.quaternion[0:3])
		R_ECEF_from_B = trimesh.transformations.quaternion_matrix(quaternion)
		Translate_COG = trimesh.transformations.translation_matrix(-assembly.COG)
		Translate_ECEF = trimesh.transformations.translation_matrix(assembly.position)

		Matrix = Translate_Large_Mass@Translate_ECEF@R_ECEF_from_B@Translate_COG

		assembly.collision.collision_handler.set_transform("Collision_"+str(assembly.id), Matrix)
		assembly.collision.original_handler.set_transform("Original_"+str(assembly.id), Matrix)

	pass

def update_collision_mesh_time(titan, options, dt):
	'''
Projects assembly collision meshes forward in time (in local ECEF frame)
	:param titan: Object of class AssemblyList
	:type titan: object
	:param options: Object of class Options
	:type options: object
	:param dt: Distance in time to project forward
	:type dt: float
'''
	mass = []
	mesh_collision = []

	for assembly in titan.assembly:
		mass.append(assembly.mass)

	index_mass = np.argmax(mass)
	dx = dt*assembly.velocity
	if hasattr(assembly, 'acceleration'): dx += 0.5*assembly.acceleration*dt**2
	Translate_Large_Mass = trimesh.transformations.translation_matrix(-(titan.assembly[index_mass].position + dx))

	for assembly in titan.assembly:
		position = deepcopy(assembly.position) + dt*assembly.velocity
		if hasattr(assembly, 'acceleration'):  position += 0.5*assembly.acceleration*dt**2
		q = assembly.quaternion
		py_quat = pyquaternion.Quaternion(q[3],q[0],q[1],q[2])
		py_quat.integrate([assembly.roll_vel, assembly.pitch_vel,assembly.yaw_vel], dt)
		quaternion = np.append( py_quat.real, py_quat.vector)
		
		R_ECEF_from_B = trimesh.transformations.quaternion_matrix(quaternion)
		Translate_COG = trimesh.transformations.translation_matrix(-assembly.COG)
		Translate_ECEF = trimesh.transformations.translation_matrix(position)

		Matrix = Translate_Large_Mass@Translate_ECEF@R_ECEF_from_B@Translate_COG

		assembly.collision.collision_handler.set_transform("Collision_"+str(assembly.id), Matrix)
	return

def generate_surface(titan, options):
	'''
Generates a debug .stl of the collision
	:param titan: Object of class AssemblyList
	:type titan: object
	:param options: Object of class Options
	:type options: object
'''
	mesh = []
	mass = []

	for assembly in titan.assembly:
		mass.append(assembly.mass)

	index_mass = np.argmax(mass)
	Translate_Large_Mass = trimesh.transformations.translation_matrix(-titan.assembly[index_mass].position)

	for assembly in titan.assembly:
		mesh_aux = deepcopy(np.sum(assembly.collision.collision_mesh))

		quaternion = np.append([assembly.quaternion[3]], assembly.quaternion[0:3])
		R_ECEF_from_B = trimesh.transformations.quaternion_matrix(quaternion)
		Translate_COG = trimesh.transformations.translation_matrix(-assembly.COG)
		Translate_ECEF = trimesh.transformations.translation_matrix(assembly.position)

		Matrix = Translate_Large_Mass@Translate_ECEF@R_ECEF_from_B@Translate_COG
		mesh_aux = mesh_aux.apply_transform(Matrix)
	
		mesh.append(mesh_aux)

	mesh = np.sum(mesh)

	mesh.export("collision_test_"+str(titan.iter)+".stl")

def generate_collision_mesh(assembly, options):
	'''
Construct a collision mesh for each assembly
	:param titan: Object of class AssemblyList
	:type titan: object
	:param options: Object of class Options
	:type options: object
'''
	assembly.collision = Collision()

	collision_mesh = []
	original_mesh = []
	factor = options.collision.mesh_factor

	for obj in assembly.objects:
		obj_collision_trimesh = generate_inflated_mesh(deepcopy(obj.mesh.nodes), deepcopy(obj.mesh.facets), factor)
		collision_mesh.append(obj_collision_trimesh)

		obj_original_trimesh = trimesh.Trimesh(vertices=obj.mesh.nodes, faces=obj.mesh.facets, process=False)
		obj_original_trimesh.fix_normals()
		original_mesh.append(obj_original_trimesh)

	assembly.collision.collision_mesh = collision_mesh
	assembly.collision.original_mesh = original_mesh

def generate_inflated_mesh(nodes, facets, factor):
	'''
"Inflates" a mesh by a factor for use in collision modelling
	:param nodes: Array of mesh node positions
	:type nodes: Any
	:param facets: Array of facet connectivity
	:type facets: Any
	:param factor: Inflation factor
	:type factor: Any

'''
	#Create a Trimesh object from the stl mesh
	collision_mesh = trimesh.Trimesh(vertices=nodes, faces=facets, process=False)

	#Fix the normals when required
	collision_mesh.fix_normals()

	#Add thickness to the mesh
	vertex_normals = collision_mesh.vertex_normals
	collision_mesh.vertices += factor*vertex_normals

	#Generate a new Trimesh object
	collision_mesh = trimesh.Trimesh(vertices=collision_mesh.vertices, faces=collision_mesh.faces, process=False)

	return collision_mesh#.convex_hull

def find_ToI_timestep(titan, options, input_time_step):
	'''
Select a time step such that no "Time-Of-Impact" points are skipped
	:param titan: Object of class AssemblyList
	:type titan: object
	:param options: Object of class Options
	:type options: object
	:param input_time_step: Maximal considered time step
	:type input_time_step: Any
'''
	#If more points of contact between assemblies exist, just the one with more depth is considered
	if len(titan.assembly) <= 1: return input_time_step
	minLref = np.min([_assembly.Lref for _assembly in titan.assembly])
	Lref_time_step = compute_time_resolution(titan, options, minLref)
	
	dt = binary_search_TOI(titan, options, input_time_step, lref_time_resolution=Lref_time_step)
	
	if dt>=input_time_step: return input_time_step
	res_time = np.min([compute_time_resolution(titan, options, 2.5e-2), 0.01])
	if options.verbose: print('Selected a dt of {}'.format(np.max([res_time, dt])))
	return np.max([res_time, dt])

def collision_physics(titan, options):
	'''
Impulsive single-collision resolution
	:param titan: Object of class AssemblyList
	:type titan: object
	:param options: Object of class Options
	:type options: object
'''
	#Restituition coeff and friction
	#u = 0.072
	#e = 0.53
	u = 0.0
	e = options.collision.elastic_factor
	collision_data = titan.collision_data

	mass = []
	for assembly in titan.assembly:
		mass.append(assembly.mass)

	index_mass = np.argmax(mass)

	for index in range(len(collision_data["assembly"])):

		i1, i2 = collision_data["assembly"][index]
		normal = collision_data["normal"][index]
		depth = collision_data["depth"][index]
		if options.verbose: print('Processing collision with depth {}'.format(depth))
		#normal = np.array([0,0,-1])
		point = collision_data["contact_point"][index] + titan.assembly[index_mass].position
		v1 = titan.assembly[i1].velocity
		v2 = titan.assembly[i2].velocity

		w1 = [titan.assembly[i1].roll_vel, titan.assembly[i1].pitch_vel, titan.assembly[i1].yaw_vel]
		w2 = [titan.assembly[i2].roll_vel, titan.assembly[i2].pitch_vel, titan.assembly[i2].yaw_vel]
		
		mass1 = titan.assembly[i1].mass 
		mass2 = titan.assembly[i2].mass

		I1 = titan.assembly[i1].inertia 
		I2 = titan.assembly[i2].inertia

		R_ECEF_from_B_1 = Rot.from_quat(titan.assembly[i1].quaternion).as_matrix()
		R_ECEF_from_B_2 = Rot.from_quat(titan.assembly[i2].quaternion).as_matrix()

		I1 = R_ECEF_from_B_1@I1@R_ECEF_from_B_1.transpose()
		I2 = R_ECEF_from_B_2@I2@R_ECEF_from_B_2.transpose()

		I1_inv = np.linalg.inv(I1)
		I2_inv = np.linalg.inv(I2)

		r1 = point-titan.assembly[i1].position
		r2 = point-titan.assembly[i2].position

		#Check if points are coming closer to each other or moving away

		Vab = v1-v2 + np.cross(R_ECEF_from_B_1@w1, point-titan.assembly[i1].position) - np.cross(R_ECEF_from_B_2@w2, point-titan.assembly[i2].position)
		Vab_dot_n = np.dot(Vab,normal)

		if Vab_dot_n <=0: 
			## Baumgarte Stabilisation to prevent intrusion
			baum_coeff = 0.00#1 # [0-1] : "Correction per second" inverse relaxation time
			slop = 1e-3       # m : allowable intersection depth
			baum_bias = (baum_coeff/titan.delta_t)*max(0,depth-slop)
			if baum_bias>0 and options.verbose: print('Baumgarte Bias Applied! Depth of {} m Bias of {} m/s'.format(depth,baum_bias))
			Vab_dot_n+=min(baum_bias, 10)
			if Vab_dot_n <=0: continue
			
		if options.verbose: print('Relative Collision Velocity of {}'.format(Vab_dot_n))

		jr = -(1+e)* Vab_dot_n/(1/mass1+1/mass2 + np.dot((np.cross(np.matmul(I1_inv,np.cross(r1,normal)),r1)+np.cross(np.matmul(I2_inv,np.cross(r2,normal)),r2)),normal))

		_v1 = v1 + jr/mass1*normal
		_v2 = v2 - jr/mass2*normal

		_w1 = w1 + Rot.from_quat(titan.assembly[i1].quaternion).inv().apply(jr*np.dot(I1_inv,np.cross(r1,normal)))
		_w2 = w2 - Rot.from_quat(titan.assembly[i2].quaternion).inv().apply(jr*np.dot(I2_inv,np.cross(r2,normal)))

		"""
		t = (Vab-(np.dot(Vab,normal)*normal))
		t_normal = t/np.linalg.norm(t)
		
		jd = u*jr
		jf = jd

		_v1 = _v1 + jf/mass1*t_normal
		_v2 = _v2 - jf/mass2*t_normal

		_w1 = _w1 + Rot.from_quat(titan.assembly[i1].quaternion).inv().apply(jf*np.dot(I1_inv,np.cross(r1,t_normal)))
		_w2 = _w2 - Rot.from_quat(titan.assembly[i2].quaternion).inv().apply(jf*np.dot(I2_inv,np.cross(r2,t_normal)))
		"""

		titan.assembly[i1].velocity = _v1
		titan.assembly[i2].velocity = _v2

		titan.assembly[i1].roll_vel  = _w1[0]
		titan.assembly[i1].pitch_vel = _w1[1]
		titan.assembly[i1].yaw_vel   = _w1[2]
		
		titan.assembly[i2].roll_vel  = _w2[0] 
		titan.assembly[i2].pitch_vel = _w2[1] 
		titan.assembly[i2].yaw_vel   = _w2[2]

		if hasattr(titan.assembly[i1], 'state_vector'):
			titan.assembly[i1].state_vector[3:6] = _v1
			titan.assembly[i1].state_vector[10:13] = _w1
		
		if hasattr(titan.assembly[i2], 'state_vector'):
			titan.assembly[i2].state_vector[3:6] = _v2
			titan.assembly[i2].state_vector[10:13] = _w2
		
		titan.collision_data = None


def collision_physics_simultaneous(titan, options):
	"""Documentation for the function.
	:param titan: TITAN simulation object.
	:type titan: object
	:param options: Options or configuration object.
	:type options: object
	:return: Return value.
	:rtype: Any"""
	#Can be improved for speed
	
	def sign(a=-1,b=-1,i = 0):
		"""Documentation for the function.
		:param a: Value for a.
		:type a: Any
		:param b: Value for b.
		:type b: Any
		:param i: Integer value for i.
		:type i: int
		:return: Return value.
		:rtype: Any"""
		if i==a: return -1
		elif i==b: return 1
		else: return 0
	u = 0
	e = 1.0
	
	collision_data = titan.collision_data

	mass = []
	for assembly in titan.assembly:
		mass.append(assembly.mass)

	index_mass = np.argmax(mass)
	number_collisions = len(collision_data["contact_point"])

	for point in collision_data["contact_point"]:
		point += titan.assembly[index_mass].position

	A = np.zeros((number_collisions, number_collisions))
	B = np.zeros((number_collisions))

	for i in range(number_collisions):
		a_i, b_i = collision_data["assembly"][i]

		ma_i = titan.assembly[a_i].mass 
		mb_i = titan.assembly[b_i].mass 

		Ia_i = titan.assembly[a_i].inertia 
		Ib_i = titan.assembly[b_i].inertia

		R_ECEF_from_B_a_i = Rot.from_quat(titan.assembly[a_i].quaternion).as_matrix()
		R_ECEF_from_B_b_i = Rot.from_quat(titan.assembly[b_i].quaternion).as_matrix()

		Ia_i = R_ECEF_from_B_a_i@Ia_i@R_ECEF_from_B_a_i.transpose() 
		Ib_i = R_ECEF_from_B_b_i@Ib_i@R_ECEF_from_B_b_i.transpose()

		Ia_i_inv = np.linalg.inv(Ia_i)
		Ib_i_inv = np.linalg.inv(Ib_i)

		r_i = collision_data["contact_point"][i]
		n_i = collision_data["normal"][i]
		#if titan.iter == 4 or titan.iter ==6: n_i = np.array([0,0,1])
		#if titan.iter ==5:
		#	if i == 0: n_i = np.array([0, -0.5, np.sqrt(3)/2])
		#	if i == 1: n_i = np.array([0, 0.5, np.sqrt(3)/2])

		#print(n_i, titan.iter)
		ra_i = titan.assembly[a_i].position
		rb_i = titan.assembly[b_i].position

		va_i = titan.assembly[a_i].velocity
		vb_i = titan.assembly[b_i].velocity

		wa_i = [titan.assembly[a_i].roll_vel, titan.assembly[a_i].pitch_vel, titan.assembly[a_i].yaw_vel]
		wb_i = [titan.assembly[b_i].roll_vel, titan.assembly[b_i].pitch_vel, titan.assembly[b_i].yaw_vel]
		
		for j in range(number_collisions):

			a_j, b_j = collision_data["assembly"][j]
			n_j = collision_data["normal"][j]
			r_j = collision_data["contact_point"][j]

			#if titan.iter == 4 or titan.iter ==6: n_j = np.array([0,0,1])
			#if titan.iter ==5:
			#	if j == 0: n_j = np.array([0, -0.5, np.sqrt(3)/2])
			#	if j == 1: n_j = np.array([0, 0.5, np.sqrt(3)/2])

			Aij_1 = sign(a_j, -1, a_i)*np.dot((1/ma_i*n_j + np.cross(Ia_i_inv@(np.cross((r_j - ra_i ),n_j )),(r_i - ra_i))),n_i)
			Aij_2 = sign(-1, b_j, b_i)*np.dot((1/mb_i*n_j + np.cross(Ib_i_inv@(np.cross((r_j - rb_i ),n_j )),(r_i - rb_i))),n_i)
			A[i,j] = Aij_1 - Aij_2

		Vab = vb_i-va_i + np.cross(R_ECEF_from_B_b_i@wb_i, r_i-rb_i )-np.cross(R_ECEF_from_B_a_i@wa_i, r_i-ra_i)
		B[i] = np.dot(Vab ,n_i) 

	P = np.linalg.solve(A,B)
	P = -(1+e)*P

	for i in range(len(P)):
		a_i, b_i = collision_data["assembly"][i]
		normal = collision_data["normal"][i]
		point = collision_data["contact_point"][i]

		n_i = normal
		#print(normal)
		#if titan.iter == 4 or titan.iter ==6: n_i = np.array([0,0,1])
		#if titan.iter ==5:
		#	if i == 0: n_i = np.array([0, -0.5, np.sqrt(3)/2])
		#	if i == 1: n_i = np.array([0, 0.5, np.sqrt(3)/2])
		normal = n_i


		ma_i = titan.assembly[a_i].mass 
		mb_i = titan.assembly[b_i].mass 
		
		ra_i = titan.assembly[a_i].position
		rb_i = titan.assembly[b_i].position

		Ra = Rot.from_quat(titan.assembly[a_i].quaternion).as_matrix()
		Rb = Rot.from_quat(titan.assembly[b_i].quaternion).as_matrix()

		Ia_i_inv = np.linalg.inv(Ra@titan.assembly[a_i].inertia@Ra.transpose()) 
		Ib_i_inv = np.linalg.inv(Rb@titan.assembly[b_i].inertia@Rb.transpose())

		R_ECEF_B_a_i = Rot.from_quat(titan.assembly[a_i].quaternion).inv()
		R_ECEF_B_b_i = Rot.from_quat(titan.assembly[b_i].quaternion).inv()

		titan.assembly[a_i].velocity += 1/ma_i*P[i]*normal
		titan.assembly[b_i].velocity -= 1/mb_i*P[i]*normal

		titan.assembly[a_i].roll_vel  += R_ECEF_B_a_i.apply(P[i]*np.dot(Ia_i_inv,np.cross(point - ra_i,normal)))[0]
		titan.assembly[a_i].pitch_vel += R_ECEF_B_a_i.apply(P[i]*np.dot(Ia_i_inv,np.cross(point - ra_i,normal)))[1]
		titan.assembly[a_i].yaw_vel   += R_ECEF_B_a_i.apply(P[i]*np.dot(Ia_i_inv,np.cross(point - ra_i,normal)))[2]

		titan.assembly[b_i].roll_vel  -= R_ECEF_B_b_i.apply(P[i]*np.dot(Ib_i_inv,np.cross(point - rb_i,normal)))[0]
		titan.assembly[b_i].pitch_vel -= R_ECEF_B_b_i.apply(P[i]*np.dot(Ib_i_inv,np.cross(point - rb_i,normal)))[1]
		titan.assembly[b_i].yaw_vel   -= R_ECEF_B_b_i.apply(P[i]*np.dot(Ib_i_inv,np.cross(point - rb_i,normal)))[2]

def binary_search_TOI(titan, options, input_dt : float, n_sanity : int | None = None, lref_time_resolution : float | None = None):
	'''
Find Time-Of-Impact through binary search of next time step
	:param titan: Object of class AssemblyList
	:type titan: object
	:param options: Object of class Options
	:type options: object
	:param input_dt: Maximal step distance
	:type input_dt: Any
	:param n_sanity: Number of sanity checks to make along the time step
	:type n_sanity: int
	:param lref_time_resolution: The duration it takes for the fastest v_rel to cross the smallest l_ref
	:type lref_time_resolution: Any
'''

	value_depth = 1
	min_time_step = 0
	max_time_step = input_dt
	dt = input_dt
	max_depth = options.collision.max_depth
	best_collision = None
	collided = False
	collided_iter = False

	max_iters = 24
	_iter = 0

	# Want also to do some quick sanity checks
	if n_sanity is not None:
		# this is N-sane!
		sanity_points = np.linspace(0,1,n_sanity+1,endpoint=False)[1:]
	else: # Unless specified auto populate our sanity checks at current dt increments
		if lref_time_resolution is None: lref_time_resolution = titan.delta_t
		sanity_points = np.arange(0,1,lref_time_resolution)[1:]
		n_sanity = len(sanity_points)
		# These checks are comparatively cheap vs true timesteps, 
		# make sense to check more often than we solve
	i_sanity_check  = 0

	while value_depth > max_depth:
		_iter+=1
		depth = []

		collided_iter = False

		flag, depth, data = update_and_check(titan, options, dt)
		if not collided: collided = flag 
		collided_iter = flag
		if _iter >= max_iters and collided_iter: break
		
		#print('Collision {} at dt={}'.format(collided_iter, dt))
		if len(depth) != 0:
			value_depth = np.max(depth)
			#print('Penetration Depth Of {}'.format(value_depth))
			if value_depth > max_depth: 
				max_time_step = dt
				dt = (dt+min_time_step)/2

			else:
				best_collision = data
				if options.verbose: print('Best collision at {}'.format(dt))
				min_time_step = dt
				dt = (max_time_step+dt)/2

		elif collided:
			min_time_step = dt
			dt = (max_time_step+dt)/2

		

		if collided==False:
			if i_sanity_check>=n_sanity: return input_dt
			else: 
				dt = (1-sanity_points[i_sanity_check])*min_time_step+sanity_points[i_sanity_check]*max_time_step
				i_sanity_check+=1

	if best_collision is not None: titan.collision_data = best_collision
	
	return dt

def update_and_check(titan, options, dt):
	'''
Update collision mesh and check for collisions
	:param titan: Object of class AssemblyList
	:type titan: object
	:param options: Object of class Options
	:type options: object
	:param dt: Time step to project forward to
	:type dt: float
'''

	#Initialize collison data dictionary
	depth = []
	collision_data = {}
	collision_data["assembly"] = []
	collision_data["names"] = []
	collision_data["index"] = []
	collision_data["contact_point"] = []
	collision_data["normal"] = []
	collision_data["depth"] = []

	#Update the collision mesh positions in future time to chek for potential collisions
	update_collision_mesh_time(titan, options, dt)
	length_assembly = len(titan.assembly) 
	i = 0
	j = 1
	depth =[]
	collided = False

	#Loop assemblies to check for collision and decide the best time_step for collision handling
	for index_i in range(i, length_assembly):
		if (index_i == length_assembly-1): break

		for index_j in range(j, length_assembly):
			if (index_j <= index_i): continue
			flag, data = titan.assembly[index_i].collision.collision_handler.in_collision_other(titan.assembly[index_j].collision.collision_handler, return_names = False, return_data = True)
			#If collision has occurred
			if flag:# or collided:
				collided = False
				new_depths = []
				for _data in data:
					# Not exactly sure what a collision with one object is but I think we want to avoid it
					if len(_data.names)>1:
						depth.append(_data._depth)
						new_depths.append(_data.depth)
						collided=True
				if collided:
					collision_data = select_contacts(data, new_depths, collision_data, [index_i, index_j])
					#ind = np.argmax(new_depths)
					# n_contacts = 50 # This is to a certain extent a tuning parameter, reccommend not going below 3 though
					# for ind in np.argsort(new_depths)[:-n_contacts-1:-1]:
					# 	collision_data["assembly"].append([index_i,index_j])
					# 	collision_data["names"].append(list(data[ind].names))
					# 	collision_data["index"].append([data[ind]._inds[collision_data["names"][-1][0]], data[ind]._inds[collision_data["names"][-1][1]]])
					# 	collision_data["contact_point"].append(data[ind]._point)
					# 	collision_data["normal"].append(data[ind]._normal)
					# 	collision_data["depth"].append(data[ind]._depth)

	return collided, depth, collision_data

def updated_fixed_contacts(titan, options, dt, collision_data):
	"""Documentation for the function.
	:param titan: TITAN simulation object.
	:type titan: object
	:param options: Options or configuration object.
	:type options: object
	:param dt: Numeric value for dt.
	:type dt: float
	:param collision_data: Value for collision data.
	:type collision_data: Any
	:return: Return value.
	:rtype: Any"""
	update_collision_mesh_time(titan, options, dt)
	for i_col in range(len(collision_data['contact_point'])):
		iA, iB = collision_data["assembly"][i_col]
		flag, data = titan.assembly[iA].collision.collision_handler.in_collision_other(titan.assembly[iB].collision.collision_handler, return_names = False, return_data = True)
		
		new_depths = []
		for _data in data: new_depths.append(_data._depth)
		if len(new_depths)>0:
			ind = np.argmax(new_depths)

			collision_data['contact_point'][i_col] = data[ind]._point
			collision_data['normal'][i_col] = data[ind]._normal
			collision_data['depth'][i_col] = data[ind]._depth
		else:
			collision_data['depth'][i_col] = -1
	return collision_data

def compute_time_resolution(titan, options, distance_resolution = 1e-6):
	'''
Maximal allowable distance error converted to a maximal time step
	:param titan: Object of class AssemblyList
	:type titan: object
	:param options: Object of class Options
	:type options: object
	:param distance_resolution: Maximal allowable distance error
	:type distance_resolution: Any
'''
	max_V = 1e-6
	for i_assembly, _assembly_A in enumerate(titan.assembly):
		for j_assembly, _assembly_B in enumerate(titan.assembly):
			if j_assembly<=i_assembly: continue
			vA = _assembly_A.velocity
			vB = _assembly_B.velocity

			vAB = np.linalg.norm(vA-vB)

			if vAB>max_V: max_V = vAB
	return distance_resolution/max_V

def select_contacts(contact_data, depths, collision_data, assem_indices):
	"""Optimises to find the best contacts
:param contact_data: Value for contact data.
:type contact_data: Any
:param depths: Value for depths.
:type depths: Any
:param collision_data: Value for collision data.
:type collision_data: Any
:param assem_indices: Value for assem indices.
:type assem_indices: Any
:return: Return value.
:rtype: Any"""
	def col_append(index):
		"""Documentation for the function.
		:param index: Integer value for index.
		:type index: int"""
		collision_data["assembly"].append([assem_indices[0],assem_indices[1]])
		collision_data["names"].append(list(contact_data[index].names))
		collision_data["index"].append([contact_data[index]._inds[collision_data["names"][-1][0]], contact_data[index]._inds[collision_data["names"][-1][1]]])
		collision_data["contact_point"].append(contact_data[index]._point)
		collision_data["normal"].append(contact_data[index]._normal)
		collision_data["depth"].append(contact_data[index]._depth)
	n_contacts = len(contact_data)
	## [1] First select the deepest contact 
	deepest_index = np.argmax(depths)
	col_append(deepest_index)
	if n_contacts<2: return collision_data

	## [2] Next the contact furthest from the deepest contact
	points_array = np.array([contact._point for contact in contact_data])
	deepest_point = points_array[deepest_index, :]
	dx = points_array - deepest_point

	furthest_from_deepest = np.argmax(np.linalg.norm(dx,axis=1))
	col_append(furthest_from_deepest)
	if n_contacts<3: return collision_data

	## [3] Next the contact furthest from the line defined by points 1 and 2
	line_norm = dx[furthest_from_deepest,:] / np.linalg.norm(dx[furthest_from_deepest, :])
	projection = dx @ line_norm
	perpendicular_dist =  np.sqrt(np.max([np.linalg.norm(dx, axis=1)**2-projection**2,np.zeros(n_contacts)],axis=0))
	furthest_from_line = np.argmax(perpendicular_dist)
	col_append(furthest_from_line)
	if n_contacts<4: return collision_data

	## [4] Finally the contact furthest from the plane defined by points 1, 2 and 3
	normal = np.cross(line_norm, dx[furthest_from_line])
	normal /= np.linalg.norm(normal)
	furthest_from_plane = np.argmax(np.abs(dx @ normal))
	col_append(furthest_from_plane)

	return collision_data
