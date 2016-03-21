import numpy as np
from scipy.sparse.linalg import *
from abc import ABCMeta
from abc import abstractmethod

class QCircuitNode:
    def __init__(self, name):
        self.name = name

class QVariable:
    def __init__(self, name):
        self.name = name
    def create_grid(self, nodeNo, phase_periods):
        minNode = np.round(-(nodeNo-1)/2)
        maxNode = np.round((nodeNo-1)/2)
        self.phase_grid = np.linspace(-np.pi*phase_periods, np.pi*phase_periods, nodeNo, endpoint=False)
        self.charge_grid = np.linspace(minNode, maxNode, nodeNo)
    def set_parameter(self, phase_value, charge_value):
        self.phase_grid = phase_value
        self.charge_grid = charge_value
    def get_phase_grid(self):
        return self.phase_grid
    def get_charge_grid(self):
        return self.charge_grid

class QCircuitElement:
    __metaclass__ = ABCMeta
    def __init__(self, name):
        self.name = name
    @abstractmethod
    def energy_term(self, node_phases, node_charges):
        pass
    @abstractmethod
    def is_phase(self):
        pass
    @abstractmethod
    def is_charge(self):
        pass
    def is_diagonal(self):
        pass
        
class QCapacitance(QCircuitElement):
    def __init__(self, name, capacitance=0):
        self.name = name
        self.capacitance = capacitance
    def set_capacitance(self, capacitance):
        self.capacitance = capacitance
    def get_capacitance(self):
        return self.capacitance
    def energy_term(self, node_phases, node_charges):
        if len(node_charges) != 2:
            raise Exception('ConnectionError', 
                            'Capacitor {0} has {1} nodes connected instead of 2.'.format(self.name, len(node_charges)))
        return (node_charges[0]-node_charges[1])**2/(2*self.capacitance)
    def is_phase(self):
        return False
    def is_charge(self):
        return True
    def is_diagonal(self):
        return True

class QJosephsonJunction(QCircuitElement):
    def __init__(self, name, critical_current=0):
        self.name = name
        self.critical_current = critical_current
    def set_critical_current(self, critical_current):
        self.critical_current = critical_current
    def get_critical_current(self):
        return self.critical_current
    def energy_term(self, node_phases, node_charges):
        if len(node_phases) != 2:
            raise Exception('ConnectionError', 
                            'Josephson junction {0} has {1} nodes connected instead of 2.'.format(self.name, len(node_phases)))
        return self.critical_current*(1-np.cos(node_phases[0]-node_phases[1]))
    def is_phase(self):
        return True
    def is_charge(self):
        return False
    def is_diagonal(self):
        return True
    
class QInductance(QCircuitElement):
    def __init__(self, name, inductance=0):
        self.name = name
        self.inductance = inductance
    def set_inductance(self, inductance):
        self.inductance = inductance
    def get_inductance(self):
        return self.inductance
    def energy_term(self, node_phases, node_charges):
        if len(node_phases) != 2:
            raise Exception('ConnectionError', 
                            'Inductance {0} has {1} nodes connected instead of 2.'.format(self.name, len(node_phases)))
        return (node_phases[0]-node_phases[1])**2/(2*self.inductance)
    def is_phase(self):
        return True
    def is_charge(self):
        return False
    def is_diagonal(self):
        return True
    
class QSubcircuit(QCircuitElement):
    def __init(self, name, subcircuit, mode):
        self.name = name
        self.subcircuit = subcircuit
        self.mode = mode
    def set_subcircuit(self, subcircuit):
        self.subcircuit = subcircuit
    def get_subcircuit(self, subcircuit):
        return self.subcircuit
    def setup_Hilbert_space(self, ranges, step_nums):
        self.Hilbert_space = Hilbert_space
        if (self.mode == 'phase'):
            pass 
        else:
            pass
    def Hamiltonian_action(self, state_vector):
        pass
    def is_phase(self):
        if (self.mode == 'phase'):
            return True
        else:
            return False
    def is_charge(self):
        if (self.mode == 'charge'):
            return True
        else:
            return False
    def is_fast(self):
        return False

class QCircuit:
    def __init__(self, tolerance=1e-18):
        self.nodes = [QCircuitNode('GND')]
        self.elements = []
        self.wires = []
        self.variables = []
        self.linear_coordinate_transform = np.asarray(0)
        self.invalidation_flag = True
        self.tolerance = tolerance
        
    def find_element(self, element_name):
        for element in self.elements:
            if element.name == element_name:
                return element
            
    def find_variable(self, variable_name):
        for variable in self.variables:
            if variable.name == variable_name:
                return variable
        
    def add_element(self, element, node_names=[]):
        self.elements.append(element)
        for node_name in node_names:
            nodes_found = 0
            for node in self.nodes:
                if (node.name == node_name):
                    self.wires.append((element.name, node.name))
                    nodes_found += 1
            if (nodes_found == 0):
                self.nodes.append(QCircuitNode(node_name))
                self.wires.append((element.name, node_name))
        self.invalidation_flag = True
        
    def add_variable(self, variable):
        self.variables.append(variable)
        self.invalidation_flag = True
        
    def map_nodes_linear(self, node_names, variable_names, coefficients):
        node_ids = []
        variable_ids = []
        for node_name in node_names:
            for node_id, node in enumerate(self.nodes):
                if node.name == node_name:
                    node_ids.append(node_id)
        for variable_name in variable_names:
            for variable_id, variable in enumerate(self.variables):
                if variable.name == variable_name:
                    variable_ids.append(variable_id)
        if len(variable_ids) != len(self.variables):
            raise Exception('VariableError', 
                            'Wrong number of variables in variable list. Got {0}, expected {1}'.format(
                                    len(variable_ids), len(self.variables)))
        if len(node_ids) != len(self.nodes):
            raise Exception('VariableError', 
                            'Wrong number of nodes in node list. Got {0}, expected {1}'.format(
                                    len(node_ids), len(self.nodes)))
        [variable_idx,node_idx] = np.meshgrid(variable_ids, node_ids)
        self.linear_coordinate_transform = coefficients[node_idx, variable_idx]
        self.invalidation_flag = True
        
    def create_phase_grid(self):
        self.invalidation_flag = True
        axes = []
        for variable in self.variables:
            axes.append(variable.get_phase_grid())
        return np.meshgrid(*tuple(axes))
        
    def create_charge_grid(self):
        self.invalidation_flag = True
        axes = []
        for variable in self.variables:
            axes.append(variable.get_charge_grid())
        return np.meshgrid(*tuple(axes))
        
    def hamiltonian_phase_action(self, state_vector):
        psi = np.reshape(state_vector, self.charge_potential.shape)
        phi = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(psi)))
        Up = self.phase_potential.ravel()*state_vector
        Tp = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(self.charge_potential*phi))).ravel()
        return Up+Tp
        
    def hamiltonian_charge_action(self, state_vector):
        phi = np.reshape(state_vector, self.charge_potential.shape)
        psi = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(phi)))
        Up = self.charge_potential.ravel()*state_vector
        Tp = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(self.phase_potential*psi))).ravel()
        return Up+Tp    
    
    def hamiltonian_setup_action_factorized(self, factor_variables, linear_factor_id, 
                                            quadratic_factors_wavefunctions_phase_f, 
                                            quadratic_factors_wavefunctions_phase_i):
        self.wavefunction_factors = factor_variables
        self.quadratic_factors_wavefunctions_phase_f = quadratic_factors_wavefunctions_phase_f
        self.quadratic_factors_wavefunctions_phase_i = quadratic_factors_wavefunctions_phase_i
        self.linear_factor_id = linear_factor_id
    
    def hamiltonian_phase_action_factorized(self, linear_factor):
        pass
        
        
    def capacitance_matrix(self):
        capacitance_matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for element in self.elements:
            if element.is_charge():
                element_node_ids = []
                for wire in self.wires:
                    if wire[0] == element.name:
                        for node_id, node in enumerate(self.nodes):
                            if wire[1]==node.name:
                                element_node_ids.append(node_id)
                if len(element_node_ids) != 2:
                    raise Exception('VariableError', 
                                    'Wrong number of ports on capacitance, expected 2, got {0}'.format(len(element_node_ids)))
                capacitance_matrix[element_node_ids[0], element_node_ids[0]] += element.get_capacitance()/2
                capacitance_matrix[element_node_ids[0], element_node_ids[1]] += -element.get_capacitance()/2
                capacitance_matrix[element_node_ids[1], element_node_ids[0]] += -element.get_capacitance()/2
                capacitance_matrix[element_node_ids[1], element_node_ids[1]] += element.get_capacitance()/2
        return capacitance_matrix
    def inverse_capacitance_matrix(self):
        E,V = np.linalg.eigh(self.capacitance_matrix())
        #print (E)
        #print (self.linear_coordinate_transform)
        #print (V)
        for eigencapacitance_id, eigencapacitance in enumerate(E):
            if np.abs(eigencapacitance)>self.tolerance:
                E[eigencapacitance_id] = 1/E[eigencapacitance_id]
        return np.einsum('mj,j,lj->ml', np.conj(V), E, V)
        #T = np.linalg.inv(np.einsum('lj,lk->kj', V, self.linear_coordinate_transform))
        #print (np.einsum('ji,j,jk->ik', np.conj(V), E, V))
        #return np.einsum('ji,j,jk->ik', np.conj(V), E, V)
        
                
    def calculate_potentials(self):    
        phase_grid = self.create_phase_grid()
        charge_grid = self.create_charge_grid()
        grid_shape = phase_grid[0].shape
        grid_size = phase_grid[0].size
        phase_potential = np.zeros(grid_shape)
        charge_potential = np.zeros(grid_shape)
        for element in self.elements:
            element_node_ids = []
            for wire in self.wires:
                if wire[0]==element.name:
                    for node_id, node in enumerate(self.nodes):
                        if wire[1]==node.name:
                            element_node_ids.append(node_id)
            phase_grid = np.reshape(np.asarray(phase_grid), (len(self.variables), grid_size))
            charge_grid = np.reshape(np.asarray(charge_grid), (len(self.variables), grid_size))
            node_phases  = np.einsum('ij,jk->ik', self.linear_coordinate_transform, phase_grid)[element_node_ids, :]
            node_charges = np.einsum('ij,jk->ik', self.linear_coordinate_transform, charge_grid)[element_node_ids,:]
            node_phases = np.reshape(node_phases, (len(element_node_ids),)+grid_shape) 
            node_charges = np.reshape(node_charges, (len(element_node_ids),)+grid_shape)
            if element.is_charge():
                charge_potential += element.energy_term(node_phases=node_phases, node_charges=node_charges)
            if element.is_phase():
                phase_potential += element.energy_term(node_phases=node_phases, node_charges=node_charges)
        #Tinv = np.linalg.pinv(self.linear_coordinate_transform)
        ECmat = 0.5*np.linalg.pinv(np.einsum('ji,jk,kl->il', self.linear_coordinate_transform, self.capacitance_matrix(), self.linear_coordinate_transform))
        self.charge_potential = np.einsum('ij,ik,kj->j', charge_grid, ECmat, charge_grid)
        self.charge_potential = np.reshape(self.charge_potential, grid_shape)
        #self.charge_potential = charge_potential
        self.phase_potential = phase_potential
        self.invalidation_flag = False
        self.hamiltonian_phase = LinearOperator((grid_size, grid_size), matvec=self.hamiltonian_phase_action)
        self.hamiltonian_charge = LinearOperator((grid_size, grid_size), matvec=self.hamiltonian_charge_action)
        return self.charge_potential, self.phase_potential
    
    def diagonalize_phase(self, num_states=2, use_sparse=True):
        energies, wavefunctions = eigs(self.hamiltonian_phase, k=num_states, which='SR')
        energy_order = np.argsort(np.real(energies))
        energies = energies[energy_order]
        wavefunctions = wavefunctions[:,energy_order]
        wavefunctions = np.reshape(wavefunctions, self.charge_potential.shape+(num_states,))
        return energies, wavefunctions