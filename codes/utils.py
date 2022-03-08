"""
Helper functions for client code
"""

import numpy as np
import hapticAPI

def cartesian2polar(cspace_point):
    """
    Covert node positions in cartesian space (x, y, z) to polar space (theta, h)
    """
    h = cspace_point[2]
    theta = np.rad2deg(np.arctan2(cspace_point[1],cspace_point[0]))
    if theta < 0.:
        theta = theta + 360. # convert to the theta range of [0, 360)
    return np.array([theta, h])

class SensingCells(object):
    def __init__(self, num_angular_cells,
                        num_height_cells,
                        init_points):
        hapticAPI.initiate_config()
        hapticAPI.show_modules()
        self.num_angular_cells = num_angular_cells
        self.num_height_cells = num_height_cells
        self.num_total_cells = self.num_height_cells*self.num_angular_cells
        self.init_points = init_points
        self.create_sensing_cells()
    
    def create_sensing_cells(self):
        """
        Create sensing cells for the artifical soft skin
        - Paramters:
            num_height_cell (int): the number of cells along the height 
            num_angular_cell (int): the number of cells along circular base
        - Returns:
        """
        self.angular_cell_bounds = np.linspace(0, 360, self.num_angular_cells+1)
        self.height_cell_bounds = np.linspace(0, 260, self.num_height_cells+1)     #[0, 13, 26, ..., 247, 260]

    def check_in_cell(self, cspace_node):
        """
        Check a node in polar coordinates (theta, h) is in which cell
        Return the index of the cell to which the node belonging 
        
        @input:
        - polar node: the polar coordinates of the given node (theta, h)
        @return:
        - the index (module id) of the cell to which the node belonging
        """

        polar_node = cartesian2polar(cspace_node)
        for idx_cell, (theta_l, theta_h) in enumerate(zip(self.angular_cell_bounds[:], self.angular_cell_bounds[1:])):
            if theta_l <= polar_node[0] <= theta_h:
                theta_idx = idx_cell
                break

        for idx_cell, (h_l, h_h) in enumerate(zip(self.height_cell_bounds[:], self.height_cell_bounds[1:])):
            if h_l <= polar_node[1] <= h_h:
                h_idx = idx_cell
                break
        
        return theta_idx + h_idx*self.num_angular_cells + 1

    def get_touched_cell(self, deformed_node_ids, deformed_node_intensities):
        """
        Get the id of touch cell, taking the input as the touch regions obtained from DeformedSkinEvent.
        This assumes that only one touch occurs at one time.

        @input:
        - deformed_node_ids: the deformed regions represented by a set of deformed node indices.
        - deformed_node_intensities: the intensity of deformation at the corresponding node index.
        @return:
        - touched_cell: the id (or index) of the cell that contact occurs.
        """
        largest_deformed_node_id = deformed_node_ids[np.argmax(deformed_node_intensities)]
        cspace_node = self.init_points[largest_deformed_node_id]
        touched_cell = self.check_in_cell(cspace_node)
        print('Alala')
        return touched_cell

    def update(self, item):
        data = item.data
        updated_deformed_intensities = list()
        updated_ids = list()
        if data['numOfDeformedNodes']>0:
            for i in range(data['numOfDeformedNodes']):
                idx_new = data['arrayOfDeformedNodes'][i]['deformedNodeID']
                displacement = data['arrayOfDeformedNodes'][i]['deformedNodeIntensity']
                updated_deformed_intensities.append(displacement)
                updated_ids.append(int(idx_new))
            updated_deformed_intensities = np.array(updated_deformed_intensities)
            # search for the sensing cell is in contact
            touched_cell = self.get_touched_cell(updated_ids, updated_deformed_intensities)
            # control vibration for the corresponding vibration module (with the same ID module) on the haptic jacket
            print('Touched cell ID: {0}'.format(touched_cell))
            hapticAPI.activate_motor(module_id=touched_cell, intensity=100, duration=2000)

if __name__== 'main':
    pass