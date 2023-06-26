import trimesh
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as Rot
import pyquaternion

class Collision():
	def __init__(self):
		self.collision_mesh = None
		self.original_mesh = None
		self.collision_handler = None
		self.original_handler =  None

def generate_collision_handler(titan, options):
	
	for assembly in titan.assembly:
		assembly.collision.collision_handler = trimesh.collision.CollisionManager()
		assembly.collision.original_handler = trimesh.collision.CollisionManager()
		assembly.collision.collision_handler.add_object("Collision_"+str(assembly.id), np.sum(assembly.collision.collision_mesh))
		assembly.collision.original_handler.add_object("Original_"+str(assembly.id), np.sum(assembly.collision.original_mesh))

def delete_collision_handler(titan, options):

	for assembly in titan.assembly:
		assembly.collision.collision_handler = None
		assembly.collision.original_handler = None

def update_collision_mesh(titan, options):

	mass = []

	for assembly in titan.assembly:
		mass.append(assembly.mass)

	index_mass = np.argmax(mass)
	Translate_Large_Mass = trimesh.transformations.translation_matrix(-titan.assembly[index_mass].position)

	for assembly in titan.assembly:
		quaternion = np.append([assembly.quaternion[3]], assembly.quaternion[0:3])
		R_B_ECEF = trimesh.transformations.quaternion_matrix(quaternion)
		Translate_COG = trimesh.transformations.translation_matrix(-assembly.COG)
		Translate_ECEF = trimesh.transformations.translation_matrix(assembly.position)

		Matrix = Translate_Large_Mass@Translate_ECEF@R_B_ECEF@Translate_COG

		assembly.collision.collision_handler.set_transform("Collision_"+str(assembly.id), Matrix)
		assembly.collision.original_handler.set_transform("Original_"+str(assembly.id), Matrix)

	pass

def update_collision_mesh_time(titan, options, dt):

	mass = []
	mesh_collision = []

	for assembly in titan.assembly:
		mass.append(assembly.mass)

	index_mass = np.argmax(mass)

	Translate_Large_Mass = trimesh.transformations.translation_matrix(-titan.assembly[index_mass].position)

	for assembly in titan.assembly:
		position = deepcopy(assembly.position) + dt*assembly.velocity
		q = assembly.quaternion
		py_quat = pyquaternion.Quaternion(q[3],q[0],q[1],q[2])
		py_quat.integrate([assembly.roll_vel, assembly.pitch_vel,assembly.yaw_vel], dt)
		quaternion = np.append( py_quat.real, py_quat.vector)
		
		R_B_ECEF = trimesh.transformations.quaternion_matrix(quaternion)
		Translate_COG = trimesh.transformations.translation_matrix(-assembly.COG)
		Translate_ECEF = trimesh.transformations.translation_matrix(position)

		Matrix = Translate_Large_Mass@Translate_ECEF@R_B_ECEF@Translate_COG

		assembly.collision.collision_handler.set_transform("Collision_"+str(assembly.id), Matrix)
		#assembly.collision.original_handler.set_transform("Original_"+str(assembly.id), Matrix)

		#mesh_collision.append(deepcopy(np.sum(assembly.collision.collision_mesh)).apply_transform(Matrix))
		#np.sum(mesh_collision).show()

	#np.sum(mesh_collision).show()

	return

def generate_surface(titan, options):
	mesh = []
	mass = []

	for assembly in titan.assembly:
		mass.append(assembly.mass)

	index_mass = np.argmax(mass)
	Translate_Large_Mass = trimesh.transformations.translation_matrix(-titan.assembly[index_mass].position)

	for assembly in titan.assembly:
		mesh_aux = deepcopy(np.sum(assembly.collision.collision_mesh))

		quaternion = np.append([assembly.quaternion[3]], assembly.quaternion[0:3])
		R_B_ECEF = trimesh.transformations.quaternion_matrix(quaternion)
		Translate_COG = trimesh.transformations.translation_matrix(-assembly.COG)
		Translate_ECEF = trimesh.transformations.translation_matrix(assembly.position)

		Matrix = Translate_Large_Mass@Translate_ECEF@R_B_ECEF@Translate_COG
		mesh_aux = mesh_aux.apply_transform(Matrix)
	
		mesh.append(mesh_aux)

	mesh = np.sum(mesh)

	#mesh.export("teste_"+str(titan.iter)+".stl")

def generate_collision_mesh(assembly, options):
	
	assembly.collision = Collision()

	collision_mesh = []
	original_mesh = []
	factor = options.collision.mesh_factor

	for obj in assembly.objects:
		obj_collision_trimesh = generate_inflated_mesh(deepcopy(obj.mesh.nodes), deepcopy(obj.mesh.facets), factor, obj.mesh.COG)
		collision_mesh.append(obj_collision_trimesh)

		obj_original_trimesh = trimesh.Trimesh(vertices=obj.mesh.nodes, faces=obj.mesh.facets, process=False)
		obj_original_trimesh.fix_normals()
		original_mesh.append(obj_original_trimesh)

	assembly.collision.collision_mesh = collision_mesh
	assembly.collision.original_mesh = original_mesh

def generate_inflated_mesh(nodes, facets, factor, COG = []):

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

def check_collision(titan, options, time_step):

	#If more points of contact between assemblies exist, just the one with more depth is considered
	if len(titan.assembly) <= 1: return False, time_step

	max_time_step = time_step
	min_time_step = 0
	dt = max_time_step
	
	value_depth = 1
	max_depth = options.collision.max_depth

	#Initialize collison data dictionary
	depth = []
	collision_data = {}
	collision_data["assembly"] = []
	collision_data["names"] = []
	collision_data["index"] = []
	collision_data["contact_point"] = []
	collision_data["normal"] = []
	collision_data["depth"] = []

	length_assembly = len(titan.assembly) 
	i = 0
	j = 1

	#update_collision_mesh_time(titan, options, min_time_step)
	
	collided = False
	collided_iter = False

	max_iters = 24
	_iter = 0

	while value_depth > max_depth:
		_iter+=1
		depth = []

		collided_iter = False
		#Update the collision mesh positions in future time to chek for potential collisions
		update_collision_mesh_time(titan, options, dt)
		if dt == 0: break

		#Loop assemblies to check for collision and decide the best time_step for collision handling
		for index_i in range(i, length_assembly):
			if (index_i == length_assembly-1): break

			for index_j in range(j, length_assembly):
				if (index_j <= index_i): continue
				flag, data = titan.assembly[index_i].collision.collision_handler.in_collision_other(titan.assembly[index_j].collision.collision_handler, return_names = False, return_data = True)
				#If collision has occurred
				if flag:# or collided:
					collided = True
					collided_iter = True

					for _data in data:
						depth.append(_data._depth)
					
		if _iter >= max_iters and collided_iter: break

		if len(depth) != 0:
			value_depth = np.max(depth)
			if value_depth > max_depth: 
				max_time_step = dt
				dt = (dt+min_time_step)/2

			else:
				min_time_step = dt
				dt = (max_time_step+dt)/2

		elif collided:
			min_time_step = dt
			dt = (max_time_step+dt)/2

		#print(value_depth, dt, collided_iter, collided)

		if value_depth == 1:
			return False, options.dynamics.time_step

	update_collision_mesh_time(titan, options, dt)

	#Loop assemblies to check for collision and decide the best time_step for collision handling
	for index_i in range(i, length_assembly):
		if (index_i == length_assembly-1): break

		for index_j in range(j, length_assembly):
			if (index_j <= index_i): continue

			flag, data = titan.assembly[index_i].collision.collision_handler.in_collision_other(titan.assembly[index_j].collision.collision_handler, return_names = False, return_data = True)

			if flag:
				depth = []
				
				for _data in data:
					depth.append(_data._depth)
				ind = np.argmax(depth)
				collision_data["assembly"].append([index_i,index_j])
				collision_data["names"].append(list(data[ind].names))
				collision_data["index"].append([data[ind]._inds[collision_data["names"][-1][0]], data[ind]._inds[collision_data["names"][-1][1]]])
				collision_data["contact_point"].append(data[ind]._point)
				collision_data["normal"].append(data[ind]._normal)
				collision_data["depth"].append(data[ind]._depth)
	#print(collision_data["names"], collision_data["assembly"], collision_data["normal"], collision_data["depth"])
	titan.collision_data = collision_data

	if len(titan.collision_data["assembly"]) != 0: 
		return True, np.max([5.0E-5, dt])
	else: 
		#print("Here")
		return False, options.dynamics.time_step

def collision_physics(titan, options):
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

		R_B_ECEF_1 = Rot.from_quat(titan.assembly[i1].quaternion).as_matrix()
		R_B_ECEF_2 = Rot.from_quat(titan.assembly[i2].quaternion).as_matrix()

		I1 = R_B_ECEF_1@I1@R_B_ECEF_1.transpose()
		I2 = R_B_ECEF_2@I2@R_B_ECEF_2.transpose()

		I1_inv = np.linalg.inv(I1)
		I2_inv = np.linalg.inv(I2)

		r1 = point-titan.assembly[i1].position
		r2 = point-titan.assembly[i2].position

		#Check if points are coming closer to each other or moving away

		Vab = v1-v2 + np.cross(R_B_ECEF_1@w1, point-titan.assembly[i1].position) - np.cross(R_B_ECEF_2@w2, point-titan.assembly[i2].position)
		#print("VAB: ", Vab); print(np.dot(Vab,normal)); print()
		if np.dot(Vab,normal) <=0: continue
		print(Vab)

		jr = -(1+e)* np.dot(Vab,normal)/(1/mass1+1/mass2 + np.dot((np.cross(np.matmul(I1_inv,np.cross(r1,normal)),r1)+np.cross(np.matmul(I2_inv,np.cross(r2,normal)),r2)),normal))

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


def collision_physics_simultaneous(titan, options):
	#Can be improved for speed
	
	def sign(a=-1,b=-1,i = 0):
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

		R_B_ECEF_a_i = Rot.from_quat(titan.assembly[a_i].quaternion).as_matrix()
		R_B_ECEF_b_i = Rot.from_quat(titan.assembly[b_i].quaternion).as_matrix()

		Ia_i = R_B_ECEF_a_i@Ia_i@R_B_ECEF_a_i.transpose() 
		Ib_i = R_B_ECEF_b_i@Ib_i@R_B_ECEF_b_i.transpose()

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

		Vab = vb_i-va_i + np.cross(R_B_ECEF_b_i@wb_i, r_i-rb_i )-np.cross(R_B_ECEF_a_i@wa_i, r_i-ra_i)
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