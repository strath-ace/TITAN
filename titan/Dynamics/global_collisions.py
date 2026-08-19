"""global_collisions module."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from ..Dynamics.collision import update_and_check, updated_fixed_contacts
from ..Dynamics.quaternion_operations import *
from copy import copy
from functools import partial
def construct_inv_mass_matrix(titan, ids):
	'''
Builds and inverts the block diagonal mass matrix of collision participants
	:param titan: Object of class AssemblyList
	:type titan: object
	:param ids: List of participating assembly indices
	:type ids: Any
'''
	Minv = np.zeros([6*len(ids),6*len(ids)])

	for i_body, assem_id in enumerate(ids):
		body = titan.assembly[assem_id]
		R_ECEF_body = Rot.from_quat(body.state_vector[6:10]).as_matrix()
		I_ECEF_inv = np.linalg.inv(R_ECEF_body @ body.inertia @ R_ECEF_body.T)

		Minv[6*i_body:6*i_body+3,6*i_body:6*i_body+3] = 1/body.mass * np.eye(3)
		Minv[6*i_body+3:6*i_body+6,6*i_body+3:6*i_body+6] = I_ECEF_inv
	return Minv
		
def construct_jacobian(local_position_ECEF, v, ids, e = 1.0, corrections = [0.0, 1e-6], titan=None, collision_data = None):
	'''
Builds the Jacobian and b vectors for restitution and stabilisation
	:param local_position_ECEF: Local frame datum position in ECEF frame
	:type local_position_ECEF: Any
	:param v: Global velocity vector
	:type v: Any
	:param ids: List of participating assembly indices
	:type ids: Any
	:param e: Coefficient of Restitution
	:type e: Any
	:param corrections: Correction parameters, [0] is the scale factor and [1] is the slop (allowable intersection)
	:type corrections: Any
	:param collision_data: Dict of collision data provided by collisions.update_and_check()
	:type collision_data: Any
	:param titan: Object of class AssemblyList
	:type titan: object
'''
	if titan is None: raise Exception('Must provide titan object!')
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
		b_physical[i_col]   =  vN -e * np.min([vN, 0.0])
		b_correction[i_col] = corrections[0] * np.max([collision_data['depth'][i_col]-corrections[1],0.0])/titan.delta_t
	
	return J, b_physical, b_correction

def construct_global_velocities(titan, ids, local_velocity_ECEF):
	'''
Builds the array of velocities for collision participants in the local ECEF frame
	:param titan: Object of class AssemblyList
	:type titan: object
	:param ids: List of participating assembly indices
	:type ids: Any
	:param local_velocity_ECEF: Local frame datum velocity in ECEF frame
	:type local_velocity_ECEF: Any
'''

	v = np.zeros(6*len(ids))
	for i_assem, assem_id in enumerate(ids):
		state = np.array(titan.assembly[assem_id].state_vector)
		v[6*i_assem   : 6*i_assem+3] = state[3:6] - np.array(local_velocity_ECEF)
		v[6*i_assem+3 : 6*i_assem+6] = Rot.from_quat(state[6:10]).apply(state[10:13])

	return v

def PGS_solve(A, b, n_iters = 100, warm_start = None, bounds =[0, np.inf], epsilon = 1e-10):
	'''
Projected (clamped) Gauss-Siedel solver
	:param A: Matrix
	:type A: Any
	:param b: Vector
	:type b: Any
	:param n_iters: Number of outer loop iterations to finish after
	:type n_iters: int
	:param warm_start: Best guess of solution vector, optional
	:type warm_start: Any
	:param bounds: Range to project values to
	:type bounds: Any
	:param epsilon: Criterion for early stopping if ‖X_[i] - X_[i-1]‖ < epsilon
	:type epsilon: Any
'''
	X = np.zeros_like(b) if warm_start is None else warm_start
	for i in range(n_iters):
		X_old = X.copy()
		for i_contact, x in enumerate(X):
			x = 1/A[i_contact, i_contact] * (b[i_contact] - np.sum(A[i_contact, :i_contact]*X[:i_contact]) -np.sum(A[i_contact, i_contact+1:]*X[i_contact+1:]))
			X[i_contact] = np.clip(x,bounds[0],bounds[1])
		if np.linalg.norm(X-X_old)<epsilon: return X
	return X

def global_collision_physics(titan, options, collision_data=None, correction_only=False, recurse=0, correction_method = 'split'):
	'''
Simultaneous collision resolutiuon using Projected Gauss Siedel to solve the LCP problem
	:param titan: Object of class AssemblyList
	:type titan: object
	:param options: Object of class Options
	:type options: object
	:param collision_data: Dict of collision data provided by collisions.update_and_check()
	:type collision_data: Any
	:param correction_only: Flag to only apply corrections instead of doing a full velocity update
	:type correction_only: Any
	:param recurse: Number of recursive corrections to apply
	:type recurse: Any
	:param correction_method: 'split'/'baumgarte'/'none' apply corrections via Split impulse or Baumgarte Stabilisation
	:type correction_method: Any
'''
	mass = []

	for assembly in titan.assembly:
		mass.append(assembly.mass)

	index_mass = np.argmax(mass)

	local_state = titan.assembly[index_mass].state_vector

	if collision_data is None: collision_data = titan.collision_data
	ids = []
	number_collisions = len(collision_data["contact_point"])
	if options.verbose: depth_dict = {}
	for i_col in range(number_collisions):
		iA, iB = collision_data["assembly"][i_col]
		ids.append(iA)
		ids.append(iB)
		if options.verbose: 
			col_str = str(iA)+':'+str(iB)
			if not col_str in list(depth_dict.keys()): depth_dict[col_str] = [collision_data['depth'][i_col]]
			else: depth_dict[col_str].append(collision_data['depth'][i_col])
	if options.verbose:
		for assemblies, depths in depth_dict.items():
			print('{} contacts between assemblies {}, max depth of {}'.format(len(depths), assemblies, np.max(depths)))
	
	ids = np.unique(ids)
	Minv = construct_inv_mass_matrix(titan, ids)
	v    = construct_global_velocities(titan, ids, local_state[3:6])

	J, b_physical, _ = construct_jacobian(local_state[:3], 
													 v, 
													 ids, 
												     options.collision.elastic_factor, 
													 [options.collision.relax_period, options.collision.slop],
													 titan,
													 collision_data)

	Minv_at_JT  = Minv @ J.T
	A = J @ Minv_at_JT
	A += np.eye(np.shape(A)[0])*1e-8 # Regularisation
	physical_impulse = PGS_solve(A,b_physical)

	v_physical = -(Minv_at_JT @ physical_impulse)
	impulse_update(options, ids, correction_method='none', correction_only=False, 
				titan=titan, v_physical = v_physical, v_corrective = None)
	v_new = construct_global_velocities(titan, ids, local_state[3:6])
	
	if correction_method is None: return
	
	J_new, _, b_correction = construct_jacobian(local_state[:3], 
													 v_new, 
													 ids, 
												     options.collision.elastic_factor, 
													 [options.collision.relax_period, options.collision.slop],
													 titan,
													 collision_data)
	
	Minv_at_JT = Minv @ J_new.T
	A  = J_new @ Minv_at_JT
	A += np.eye(np.shape(A)[0])*1e-8
	
	corrective_impulse = PGS_solve(A, b_correction)
	v_corrective = -(Minv_at_JT @ corrective_impulse)

	impulse_update(options, ids, correction_method=correction_method, correction_only=True,
				   titan=titan, v_corrective = v_corrective)

	if recurse>0:
		has_collided, _, collision_data = update_and_check(titan, options, 0)
		if has_collided: global_collision_physics(titan, 
												  options, 
												  collision_data, 
												  correction_only=True, 
												  recurse=recurse-1,
												  correction_method=correction_method)

def impulse_update(options, ids, correction_method='split', correction_only=False, titan=None, v_physical = None, v_corrective = None):
	"""Documentation for the function.
:param options: Options or configuration object.
:type options: object
:param ids: Value for ids.
:type ids: Any
:param correction_method: Value for correction method.
:type correction_method: Any
:param correction_only: Value for correction only.
:type correction_only: Any
:param titan: TITAN simulation object.
:type titan: object
:param v_physical: Value for v physical.
:type v_physical: Any
:param v_corrective: Value for v corrective.
:type v_corrective: Any
:return: Return value.
:rtype: Any"""
	for i_body, assem_index in enumerate(ids):
		body = titan.assembly[assem_index]
		R_body = Rot.from_quat(body.state_vector[6:10])
		if not correction_only:
			if v_physical is None: raise Exception('Must provide impulses!')
			if options.verbose: print('Changing velocity of assembly {} by {}'.format(assem_index,v_physical[6*i_body:6*i_body+3]))
			body.state_vector[3:6]   += v_physical[6*i_body:6*i_body+3]
			body.state_vector[10:13] += R_body.inv().apply(v_physical[6*i_body+3:6*i_body+6])

		match correction_method.lower():
			case 'baumgarte':
				if v_corrective is None: raise Exception('Must provide impulses!')

				body.state_vector[3:6]   += v_corrective[6*i_body:6*i_body+3]
				body.state_vector[10:13] += R_body.inv().apply(v_corrective[6*i_body+3:6*i_body+6])

			case 'split':
				if v_corrective is None: raise Exception('Must provide impulses!')

				dx = titan.delta_t * v_corrective[6*i_body:6*i_body+3]
				if options.verbose and np.any(abs(dx)>0): 
					print('Correcting position of assembly {} by {}'.format(assem_index,dx))
				body.state_vector[:3] += dx
				omega_q = np.array([v_corrective[6*i_body+3], v_corrective[6*i_body+4], v_corrective[6*i_body+5], 0])
				dq = 0.5 * quaternion_mult(body.state_vector[6:10], omega_q)
				body.state_vector[6:10] += dq * titan.delta_t
				body.state_vector[6:10] = quaternion_normalize(body.state_vector[6:10])
		
		body.position = np.array(body.state_vector[:3])
		body.velocity = np.array(body.state_vector[3:6])
		body.quaternion = np.array(body.state_vector[6:10])
		body.rol_vel, body.pitch_vel, body.yaw_vel = body.state_vector[10:13]
	return titan

def sequential_collision_resolution(titan, options, imp_update_func, MInv, J_func, b_shape, n_iters = 100, bounds =[0, np.inf], epsilon = 1e-10, warm_start = None):
	'''
PGS with contact recomputation in-loop
	:param A: Matrix
	:type A: Any
	:param b: Vector
	:type b: Any
	:param n_iters: Number of outer loop iterations to finish after
	:type n_iters: int
	:param warm_start: Best guess of solution vector, optional
	:type warm_start: Any
	:param bounds: Range to project values to
	:type bounds: Any
	:param epsilon: Criterion for early stopping if ‖X_[i] - X_[i-1]‖ < epsilon
	:type epsilon: Any
'''
	
	
	#impulses_phys = np.zeros_like(v) if warm_start is None else warm_start[0]

	col_data = titan.collision_data
	impulses_corr = np.zeros(len(col_data['contact_point'])) if warm_start is None else warm_start[1]
	v_corr = None

	for i in range(n_iters):
		old_depth = np.max(col_data['depth'])
		
		_, _, col_data = update_and_check(titan, options, 0)

		while not len(impulses_corr)==len(col_data['contact_point']):
			if len(impulses_corr)<len(col_data['contact_point']):
				impulses_corr = np.hstack([impulses_corr, 0])
			else: impulses_corr = impulses_corr[:-1]
		
		for i_v, imp_corr in enumerate(impulses_corr):
			#Recompute J
			J, _, b_corr = J_func(titan, col_data)
			Minv_at_JT = MInv @ J.T
			A = J @ Minv_at_JT
			A += np.eye(np.shape(A)[0])*1e-8
			#b = - (J @ v + b_phys)
			
			imp_corr = 1/A[i_v, i_v] * (b_corr[i_v] - np.sum(A[i_v, :i_v]*impulses_corr[:i_v]) -np.sum(A[i_v, i_v+1:]*impulses_corr[i_v+1:]))

			impulses_corr[i_v] = np.clip(imp_corr,bounds[0],bounds[1])

			#v_phys = Minv_at_JT@impulses_phys
			if v_corr is None: v_corr = Minv_at_JT@impulses_corr
			else: v_corr += Minv_at_JT@impulses_corr

			titan = imp_update_func(titan, None, v_corr)
			col_data = updated_fixed_contacts(titan, options, 0, col_data)
		if abs(np.max(col_data['depth'])-old_depth)<epsilon: return impulses_corr
	return impulses_corr
