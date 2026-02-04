import numpy as np
from scipy.spatial.transform import Rotation as Rot
from Dynamics.collision import update_and_check
from Dynamics.quaternion_operations import *
def construct_inv_mass_matrix(titan, ids):
	'''
	Builds and inverts the block diagonal mass matrix of collision participants
	
	:param titan: Object of class AssemblyList
	:param ids: List of participating assembly indices
	'''
	Minv = np.zeros([6*len(ids),6*len(ids)])

	for i_body, assem_id in enumerate(ids):
		body = titan.assembly[assem_id]
		R_ECEF_body = Rot.from_quat(body.state_vector[6:10]).as_matrix()
		I_ECEF_inv = np.linalg.inv(R_ECEF_body @ body.inertia @ R_ECEF_body.T)

		Minv[6*i_body:6*i_body+3,6*i_body:6*i_body+3] = 1/body.mass * np.eye(3)
		Minv[6*i_body+3:6*i_body+6,6*i_body+3:6*i_body+6] = I_ECEF_inv
	return Minv
		
def construct_jacobian(titan, local_position_ECEF, v, ids, e = 1.0, corrections = [0.0, 1e-6], collision_data = None):
	'''
	Builds the Jacobian and b vectors for restitution and stabilisation
	
	:param titan: Object of class AssemblyList
	:param local_position_ECEF: Local frame datum position in ECEF frame
	:param v: Global velocity vector
	:param ids: List of participating assembly indices
	:param e: Coefficient of Restitution
	:param corrections: Correction parameters, [0] is the scale factor and [1] is the slop (allowable intersection)
	:param collision_data: Dict of collision data provided by collisions.update_and_check()
	'''

	if collision_data is None: collision_data = titan.collision_data
	n_contacts = len(collision_data["contact_point"])
	J = np.zeros([n_contacts, 6*len(ids)])
	b_physical   = np.zeros([n_contacts])
	b_correction = np.zeros([n_contacts])

	for i_col in range(n_contacts):
		idA, idB = collision_data["assembly"][i_col]

		iA = list(ids).index(idA)
		iB = list(ids).index(idB)
		
		p = collision_data["contact_point"][i_col] + local_position_ECEF
		n = collision_data["normal"][i_col]
		R_ECEF_body_A = Rot.from_quat(titan.assembly[idA].state_vector[6:10])
		CoM_base_A = titan.assembly[idA].state_vector[:3]
		rA = p - CoM_base_A
		
		R_ECEF_body_B = Rot.from_quat(titan.assembly[idB].state_vector[6:10])
		CoM_base_B = titan.assembly[idB].state_vector[:3]
		rB = p - CoM_base_B

		J[i_col, 6*iA   : 6*iA+3] =  n
		J[i_col, 6*iA+3 : 6*iA+6] =  np.cross(rA,n)
		J[i_col, 6*iB   : 6*iB+3] = -n
		J[i_col, 6*iB+3 : 6*iB+6] = -np.cross(rB,n)

		vN = J[i_col,:] @ v
		b_physical[i_col]   = -e * np.min([vN, 0.0])
		b_correction[i_col] = corrections[0] * np.max([collision_data['depth'][i_col]-corrections[1],0.0])
	
	return J, b_physical, b_correction

def construct_global_velocities(titan, ids, local_velocity_ECEF):
	'''
	Builds the array of velocities for collision participants in the local ECEF frame
	
	:param titan: Object of class AssemblyList
	:param ids: List of participating assembly indices
	:param local_velocity_ECEF: Local frame datum velocity in ECEF frame
	'''

	v = np.zeros(6*len(ids))
	for i_assem, assem_id in enumerate(ids):
		state = np.array(titan.assembly[assem_id].state_vector)
		v[6*i_assem   : 6*i_assem+3] = state[3:6] - np.array(local_velocity_ECEF)
		v[6*i_assem+3 : 6*i_assem+6] = Rot.from_quat(state[6:10]).apply(state[10:])

	return v

def PGS_solve(A, b, n_iters = 100, warm_start = None, bounds =[0, np.inf], epsilon = 1e-10):
	'''
	Projected (clamped) Gauss-Siedel solver
	
	:param A: Matrix
	:param b: Vector
	:param n_iters: Number of outer loop iterations to finish after
	:param warm_start: Best guess of solution vector, optional
	:param bounds: Range to project values to
	:param epsilon: Criterion for early stopping if ‖X_[i] - X_[i-1]‖ < epsilon
	'''
	X = np.zeros_like(b) if warm_start is None else warm_start
	for i in range(n_iters):
		X_old = X.copy()
		for i_x, x in enumerate(X):
			x = 1/A[i_x, i_x] * (b[i_x] - np.sum(A[i_x, :i_x]*X[:i_x]) -np.sum(A[i_x, i_x+1:]*X[i_x+1:]))
			X[i_x] = np.clip(x,bounds[0],bounds[1])
		if np.linalg.norm(X-X_old)<epsilon: return X
	return X

def global_collision_physics(titan, options, collision_data=None, correction_only=False, recurse=0, correction_method = 'split'):
	'''
	Simultaneous collision resolutiuon using Projected Gauss Siedel to solve the LCP problem
	
	:param titan: Object of class AssemblyList
	:param options: Object of class Options
	:param collision_data: Dict of collision data provided by collisions.update_and_check()
	:param correction_only: Flag to only apply corrections instead of doing a full velocity update
	:param recurse: Number of recursive corrections to apply
	:param correction_method: 'split'/'baumgarte'/'none' apply corrections via Split impulse or Baumgarte Stabilisation
	'''
	mass = []

	for assembly in titan.assembly:
		mass.append(assembly.mass)

	index_mass = np.argmax(mass)

	local_state = titan.assembly[index_mass].state_vector

	if collision_data is None: collision_data = titan.collision_data
	ids = []
	number_collisions = len(collision_data["contact_point"])
	
	for i_col in range(number_collisions):
		iA, iB = collision_data["assembly"][i_col]
		ids.append(iA)
		ids.append(iB)
		if options.verbose: 
			print('Collision depth of {} between assemblies {} and {}'.format(collision_data['depth'][i_col],iA,iB))
	
	ids = np.unique(ids)
	Minv = construct_inv_mass_matrix(titan, ids)
	v    = construct_global_velocities(titan, ids, local_state[3:6])
	J, b_physical, b_correction = construct_jacobian(titan, 
												     local_state[:3], 
													 v, 
													 ids, 
												     options.collision.elastic_factor, 
													 [options.collision.relax_period, options.collision.slop],
													 collision_data)
	
	if np.all(b_correction<=0) and correction_only: return

	Minv_at_JT  = Minv @ J.T
	A = J @ Minv_at_JT
	A += np.eye(np.shape(A)[0])*1e-8 # Regularisation
	b = - (J @ v + b_physical)
	physical_impulse = PGS_solve(A,b)

	corrective_impulse = PGS_solve(A, b_correction)

	v_physical = Minv_at_JT @ physical_impulse

	v_corrective = Minv_at_JT @ corrective_impulse

	for i_body, assem_index in enumerate(ids):

		body = titan.assembly[assem_index]
		R_body = Rot.from_quat(body.state_vector[6:10])
		if not correction_only:
			body.state_vector[3:6]   += v_physical[6*i_body:6*i_body+3]
			body.state_vector[10:13] += R_body.inv().apply(v_physical[6*i_body+3:6*i_body+6])

		match correction_method.lower():
			case 'none': 
				if recurse==0: return
			
			case 'baumgarte':
				body.state_vector[3:6]   += v_corrective[6*i_body:6*i_body+3]
				body.state_vector[10:13] += R_body.inv().apply(v_corrective[6*i_body+3:6*i_body+6])

			case 'split':	
				dx = titan.delta_t * v_corrective[6*i_body:6*i_body+3]
				if options.verbose and np.any(abs(dx)>0): 
					print('Correcting position of assembly {} by {}'.format(assem_index,dx))
				body.state_vector[:3] += dx
				omega_q = np.array([v_corrective[6*i_body+3], v_corrective[6*i_body+4], v_corrective[6*i_body+5], 0])
				dq = 0.5 * quaternion_mult(body.state_vector[6:10], omega_q)
				body.state_vector[6:10] += dq * titan.delta_t
				

		body.position = np.array(body.state_vector[:3])
		body.velocity = np.array(body.state_vector[3:6])
		body.quaternion = np.array(body.state_vector[6:10])
		body.rol_vel, body.pitch_vel, body.yaw_vel = body.state_vector[10:]

	if recurse>0:
		has_collided, _, collision_data = update_and_check(titan, options, 0)
		if has_collided: global_collision_physics(titan, 
												  options, 
												  collision_data, 
												  correction_only=True, 
												  recurse=recurse-1,
												  correction_method=options.collision.stabilisation)