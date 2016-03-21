import numpy as np
import sys
import math
import time
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay

class AdaptiveParametricSpaceMapper:
    def __init__(self, parameters, target, error_norm):
        self.parameters = parameters
        self.target = target
        self.vertices = np.empty((0, len(parameters)), dtype=float)
        self.funvals = []
        self.error_norm = error_norm
        self.create_initial_mesh()
        self.iteration_times = []
    
    def rescale_parameters(self, values):
        rescaled = np.ndarray(values.shape)
        for parameter_id in range(len(self.parameters)):
            pmin = self.parameters[parameter_id][1]
            pmax = self.parameters[parameter_id][2]
            rescaled[parameter_id] = pmin+(pmax-pmin)*values[parameter_id]
        return rescaled
    
    def rescale_parameters_multiple(self, values):
        rescaled = np.ndarray(values.shape)
        for parameter_id in range(len(self.parameters)):
            pmin = self.parameters[parameter_id][1]
            pmax = self.parameters[parameter_id][2]
            rescaled[:,parameter_id] = pmin+(pmax-pmin)*values[:,parameter_id]
        return rescaled
    
    def inverse_rescale_parameters_multiple(self, values):
        rescaled = np.ndarray(values.shape)
        for parameter_id in range(len(self.parameters)):
            pmin = self.parameters[parameter_id][1]
            pmax = self.parameters[parameter_id][2]
            rescaled[:,parameter_id] = (values[:,parameter_id]-pmin)/(pmax-pmin)
        return rescaled
    
    # initial mesh is a n-parallelipiped
    def create_initial_mesh(self):
        self.vertices = np.empty((2**len(self.parameters)+1,len(self.parameters)), dtype=np.float)
        for vertex_id in range(2**len(self.parameters)):
            for parameter_id in range(len(self.parameters)):
                if vertex_id % (2**(parameter_id+1)) >= 2**parameter_id:
                    self.vertices[vertex_id, parameter_id] = 1
                else:
                    self.vertices[vertex_id, parameter_id] = 0
        self.vertices[2**len(self.parameters), :] = np.mean(self.vertices[:2**len(self.parameters),:], axis=0)
        #grids1d = []
        #for parameter_id in range(len(self.parameters)):
        #    grids1d.append(np.linspace(0, 1, 3))
        #gridsnd = np.asarray(np.meshgrid(*(tuple(grids1d))))
        #gridsnd = np.reshape(gridsnd, (len(self.parameters), 3**len(self.parameters)))
        #self.vertices = gridsnd.T
        self.triangulation = Delaunay(np.asarray(self.vertices), incremental=True)
        self.funvals = []
    
    def error_estimator(self):
        nsimplex = self.triangulation.simplices.shape[0]
        simplex_error = np.empty((nsimplex,))
        for simplex_id in range(nsimplex):
            simplex_points = self.triangulation.simplices[simplex_id, :]
            simplex_funvals = []
            for simplex_point in simplex_points:
                simplex_funvals.append(self.funvals[simplex_point])
            simplex_mean_funval = np.mean(simplex_funvals, axis=0)
            simplex_dfs = simplex_mean_funval - simplex_funvals
            print(simplex_dfs)
            simplex_dfs_norms = self.error_norm(simplex_dfs)
            simplex_error[simplex_id] = max(simplex_dfs_norms)
        return simplex_error
        
    def add_vertex(self, vertex):
        self.vertices.resize((self.vertices.shape[0]+1, self.vertices.shape[1]))
        self.vertices[self.vertices.shape[0]-1, :] = vertex
        self.triangulation.add_points(np.reshape(vertex, (1, len(vertex))))
    
    def run(self, max_vertices=sys.maxsize):
        while len(self.vertices)<max_vertices:
            iteration_begin = time.time()
            if (self.vertices.shape[0] != len(self.funvals)):
                self.funvals.append(self.target(self.rescale_parameters(self.vertices[len(self.funvals)-1,:])))
                iteration_end = time.time()
                self.iteration_times.append({'target_evaluation': iteration_end-iteration_begin})
            else:
                dfs = self.error_estimator()
                error_estimator_time = time.time()
                max_simplices_ids = np.argwhere(dfs==np.max(dfs))
                max_simplices_measures = np.empty(max_simplices_ids.shape)
                print((np.argmax(dfs),np.max(dfs)))
                #max_simplices_centroids = np.empty((max_simplices_ids.size, len(self.parameters)))
                for max_simplex_id, simplex_id in enumerate(max_simplices_ids):
                    simplex_vertices = self.triangulation.simplices[simplex_id, :]
                    simplex_vertices_coordinates = self.triangulation.points[simplex_vertices,:]
                    max_simplices_measures[max_simplex_id] = np.linalg.det(simplex_vertices_coordinates[0,1:,:]-simplex_vertices_coordinates[0,0,:])/math.factorial(len(self.parameters))
                max_simplex_vertices = self.triangulation.simplices[max_simplices_ids[np.argmax(max_simplices_measures)],:].ravel()
                simplex_edge_lengths = np.empty((len(max_simplex_vertices), len(max_simplex_vertices)))
                for simplex_vertex1_id, simplex_vertex1 in enumerate(max_simplex_vertices):
                    for simplex_vertex2_id, simplex_vertex2 in enumerate(max_simplex_vertices):
                        simplex_edge_lengths[simplex_vertex1_id, simplex_vertex2_id] = np.linalg.norm(self.triangulation.points[simplex_vertex1,:] - self.triangulation.points[simplex_vertex2,:])
                    print (self.rescale_parameters(self.triangulation.points[simplex_vertex1]))
                    print (np.real((self.funvals[simplex_vertex1][1]-self.funvals[simplex_vertex1][0])))
                simplex_vertex1_id, simplex_vertex2_id = np.unravel_index([np.argmax(simplex_edge_lengths.ravel())], simplex_edge_lengths.shape)
                simplex_vertex1 = max_simplex_vertices[simplex_vertex1_id]
                simplex_vertex2 = max_simplex_vertices[simplex_vertex2_id]
                simplex_chooser_time = time.time()
                #print ((simplex_vertex1[0], simplex_vertex2[0], simplex_edge_lengths[simplex_vertex1_id, simplex_vertex2_id], np.max(dfs)))
                median = 0.5*(self.triangulation.points[simplex_vertex1[0],:] + self.triangulation.points[simplex_vertex2[0],:])
                self.add_vertex(median)

                self.funvals.append(self.target(self.rescale_parameters(median)))
                iteration_end_time = time.time()
                centroid = np.mean(self.triangulation.points[max_simplex_vertices,:], axis=0)
                self.add_vertex(centroid)
                self.funvals.append(self.target(self.rescale_parameters(centroid)))
                self.iteration_times.append({'error_estimator': -iteration_begin + error_estimator_time,
                                             'simplex_chooser': -error_estimator_time + simplex_chooser_time,
                                             'target_evaluation': iteration_end_time - error_estimator_time})
