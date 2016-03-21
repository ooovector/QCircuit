# QCircuit
A simple Python module to calculate energy levels of superconducting qubits.

In order to define a circuit, the following steps should be performed:
  0. All loops should be broken apart. For that, a tree must be selected. One end of each branch should be connected to the tree. All resulting nodes should be named. One of the nodes should be called 'GND'. 
  1. Create a QCircuit object.
  2. The library defines three types of elements: capacitors, inductances and Josephson junctions. Each of these elements should be connected to two distinct nodes of the circuit. This is done with the QCircuit.add_element() function.
  3. The phase variables used to described the quantum state of the circuit (i.e. the wavefunction variables) should be decleared by instantizing the QVariable class. External controls, such as gate voltages and external fluxes should be also assinged a variable. The variables should be added to the QCircuit object.
  4. A discrete grid should be defined for each of the wavefunction variables with the QCircuit.create_grid() function.
  5. The variables should be mapped onto the nodes. This is done with QCircuit.map_nodes_linear() function. Only linear maps 
are supported. The third argument defnes the linear transform matrix from the variables to the nodes. The correspondence of the rows and columns of the matrix is given by the order of the varibles and nodes in the first two arguments.
  6. The critical currents of the josephson junctions, capacitances of the capacitors and inductances of the inductances are set.
  7. External variables are set with the QVariable.set_parameter() function. The first argument is the external phase difference (implements external fluxes), and the second one is the external charge (implements external voltages).
  8. The potential and kinetic energy landscapes should be evaluated with the QCircuit.calculate_potentials() function.
  9. The energy levels and wavefunctions can be obtained by sparse diaglonalization with the QCircuit.diagonalize_phase() function.
  
For examples, see the transmon and 3JJ flux qubit ipython notebooks.
