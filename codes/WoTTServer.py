#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import logging
import tornado.gen
from tornado.ioloop import IOLoop
from wotpy.protocols.http.server import HTTPServer
from wotpy.protocols.ws.server import WebsocketServer
from wotpy.wot.servient import Servient
import pandas as pd
import time
import asyncio
import numpy as np

import cv2
import os
import torch
from torchvision import transforms

# import TacNet
from tacnet_model import TacNet

CATALOGUE_PORT = 9090
WEBSOCKET_PORT = 9393
HTTP_PORT = 9494

logging.basicConfig()
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)


# Load TD-json file
tdJSON = open("./TDs/tactilesensor.td.json")
TD = json.load(tdJSON)
tdJSON.close()


def get_free_node_ind(node_idx_path, label_idx_path):
    df_node_idx = pd.read_csv(node_idx_path)
    df_label_idx = pd.read_csv(label_idx_path)

    node_idx = np.array(df_node_idx.iloc[:,0], dtype=int) # (full skin) face node indices in vtk file exported from SOFA 
    node_idx = list(set(node_idx)) # eleminate duplicate elements (indices)
    node_idx = sorted(node_idx) # sorted the list of indices

    label_idx = list(df_label_idx.iloc[:,0]) #(not full skin) at nodes used for training - labels
    file_idx = [node_idx.index(idx) for idx in label_idx]

    return file_idx

class TouchInformationProcessing(object):
    def __init__(self, tacnet_dir='./resources',
                       trained_model = 'TacNet_Unet_real_data.pt',
                       cam_ind = [0, 2],
                       num_of_nodes = 707,
                       node_idx_path='./resources/node_idx.csv', 
                       label_idx_path='./resources/label_idx.csv'):

        # Soft skin representation
        self.num_of_nodes = num_of_nodes
        self.free_node_ind = get_free_node_ind(node_idx_path, label_idx_path)

        # Initialize TacNet
        self.model_dir = os.path.join(tacnet_dir, trained_model)
        self.init_TacNet()

        # Initialize Cameras
        """
        For sucessfully read two video camera streams simultaneously,
        we need to use two seperated USB bus for cameras
        e.g, USB ports in front and back of the CPU
        """
        self.cam_bot = cv2.VideoCapture(cam_ind[0])
        self.cam_top = cv2.VideoCapture(cam_ind[1])
        if self.cam_bot.isOpened() and self.cam_top.isOpened():
            print('Cameras are ready!')
        else:
            assert False, 'Camera connection failed!'

    def init_TacNet(self):
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tacnet = TacNet()
        print('[Tacnet] model was created')
        self.print_networks(False)
        print('loading the model from {0}'.format(self.model_dir))
        self.tacnet.load_state_dict(torch.load(self.model_dir))
        print('---------- TacNet initialized -------------')
        self.tacnet.to(self.dev)
        self.tacnet.eval()

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        net = getattr(self, 'tacnet')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        if verbose:
            print(net)
        print('[TacNet] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    def estimate_skin_deformation(self):
        """
        Return the estimate of skin node's displacements X - X0 (N, 3)
        N is the number of nodes, and 3 is Dx, Dy, Dz
        Refer to Section 3. (Skin Deformation Estimation) in the paper
        """
        self.update_tactile_images()
        self._free_node_displacments = self.estimate_free_node_displacments()
        return self.get_full_node_displacments()

    def update_tactile_images(self):
        # Read marker-featured tactile images from camera video streams
        frame_top = cv2.cvtColor(self.cam_top.read()[1], cv2.COLOR_BGR2RGB)
        frame_bot = cv2.cvtColor(self.cam_bot.read()[1], cv2.COLOR_BGR2RGB)
        # Apply pre-processing to the pair of tactile images
        self._frame_top = self.apply_transform(frame_top)
        self._frame_bot = self.apply_transform(frame_bot)
        # Concantenate the two tactile images
        self._tac_img = torch.cat((self._frame_top, self._frame_bot), dim=1).to(self.dev)

    def estimate_free_node_displacments(self):
        with torch.no_grad():
            node_displacments = self.tacnet(self._tac_img).cpu().numpy()
            return node_displacments

    def get_full_node_displacments(self):
        """
        The full skin deformation includes deviations of fixed nodes which is zero, 
        and the free nodes calculated in "estimate" function
        """
        
        displacements = np.zeros((self.num_of_nodes, 3))
        displacements[self.free_node_ind, :] = self._free_node_displacments.reshape(-1, 3)

        return displacements

    def apply_transform(self, img):
        """
        Apply pre-processing for the inputted image
        Parameters:
            img: image in numpy nd.array (C, H, W)
        Returns:
            processed image in tensor format (1, C, H, W)
        """
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(480),
        transforms.Resize((256,256))
        ])
        return transform(img).unsqueeze(0)

    """
    Sensor Processing Method for Events
    """
    def extract_contact_area(self):
        """
        Return node indices where contact is made,
        and the corresponding node depth (displacement intensity at the node)
        """
        full_node_displacements = self.estimate_skin_deformation()
        nodes_depth = np.linalg.norm(full_node_displacements, axis=1)
        touched_nodes_indices = np.where(nodes_depth > 2.5)[0]
        return nodes_depth[touched_nodes_indices], touched_nodes_indices

    def detect_touch(self):
        """
        Trigger an event by True signal when an contact occurs on the skin
        Binary classification task
        """
        full_node_displacements = self.estimate_skin_deformation()
        nodes_depth = np.linalg.norm(full_node_displacements, axis=1)
        # extract nodes where depth > epsilon = 5 mm
        touched_nodes_depth = nodes_depth[(nodes_depth>1.5)]
        # the number of touched nodes
        num_of_touched_nodes = len(touched_nodes_depth)
        return True if num_of_touched_nodes > 2 else False

touch_processing = TouchInformationProcessing()


# async def read_shape_hanlder():
#     return {
#         'baseRadiusDimension': 90,
#         'heightDimension': 300
#     }

"""
Utils for handling resources
"""

def extract_initial_node_locations(vtk_file='./resources/skin.vtk'):
    points_ls = []
    with open(vtk_file, 'r') as rf:
        for idx, line in enumerate(rf):
            if 6 <= idx <= 712:
                points = [float(x) for x in line.split()]
                points_ls.append(points)
            elif idx > 712:
                break
            
    return np.array(points_ls, dtype=float)

def extract_skin_cells(vtk_file='./resources/skin.vtk'):
    """
    Returns:
        - list of connected nodes : [[id1, id2, id3], [], ..., []]
    """

    cells_ls = []
    with open(vtk_file, 'r') as rf:
        for idx, line in enumerate(rf):
            if idx < 715:
                continue
            elif 715 <= idx <= 2088:
                cells = line.split()
                cells_ls.append(cells)
            elif idx > 2088:
                break

    return cells_ls

def load_resources(vtk_file='./resources/skin.vtk'):
    init_points = extract_initial_node_locations(vtk_file)
    skin_cells = extract_skin_cells(vtk_file)
    return init_points, skin_cells

"""
Extract TD properties from resrouces
"""

def createSkinNodesProperty(init_points):
    # nodes' location extracted from VTK file
    nodeLocations = np.array(init_points)
    numOfNodes = nodeLocations.shape[0]
    arrayOfNodes = []
    for i in range(numOfNodes):
        nodeLocation = nodeLocations[i]
        nodeDict = {
            'nodeID': str(i),
            'nodeLocation': {
                'x': nodeLocation[0],
                'y': nodeLocation[1],
                'z': nodeLocation[2]
            },
            'referenceTo': 'world'
        }
        arrayOfNodes.append(nodeDict)
    
    return {'numOfNodes': numOfNodes, 'arrayOfNodes': arrayOfNodes}

def createSkinCellsProperty(skin_cells):
    skinCells = list(skin_cells)
    numOfCells = len(skinCells)
    arrayOfCells = []
    for i in range(numOfCells):
        skinCell = skinCells[i]
        numOfConnectedNodes = int(skinCell[0])
        ConnectedNodes = skinCell[1:]
        cellDict = {
            'numOfConnectedNodes': numOfConnectedNodes,
            'connectedNodes': ConnectedNodes
        }
        arrayOfCells.append(cellDict)
    
    return {'numOfCells': numOfCells, 'arrayOfCells': arrayOfCells}


"""
Create tasks for TD events
"""

def create_touch_detected_task(exposed_thing):
    """Inform when touch occurs."""
    async def send_event(exposed_thing):
        while True:
            #t1
            # timens = time.time_ns()
            touch_detected = touch_processing.detect_touch()
            #t2
            timens = time.time_ns()
            if touch_detected:
                alert = str(timens)
                print('Sent: {0}'.format(alert))
                #t3
                exposed_thing.emit_event('actionDetection', alert)
            await asyncio.sleep(0.05)

    event_loop = asyncio.get_event_loop()
    event_loop.create_task(send_event(exposed_thing))

def create_skin_state_update_task(exposed_thing):
    """Update node locations where touch is made (change in node locations)."""
    async def send_skin_state(exposed_thing):
        last_indices = list()
        sent = False
        while True:
            await asyncio.sleep(0.05)
            #t1
            # timens = time.time_ns()
            deformedNodeIntensities, indices = touch_processing.extract_contact_area()
            #t2
            timens = time.time_ns()
            numOfDeformedNodes = len(indices)
            arrayOfDeformedNodes = []
            if numOfDeformedNodes > 0:
                sent = True
                for i in range(numOfDeformedNodes):
                    id = indices[i]
                    deformednodeIntensity = deformedNodeIntensities[i]
                    deformedNodesDict = {
                        'deformedNodeID': str(id),
                        'deformedNodeIntensity': deformednodeIntensity,
                    }
                    arrayOfDeformedNodes.append(deformedNodesDict)
                last_indices =  indices   
                LOGGER.info('Skin Deformed!')
                #t3
                timens = time.time_ns()
                exposed_thing.emit_event('skinDeformedDetection', {'timeStamp': timens, 'numOfDeformedNodes': numOfDeformedNodes, 'arrayOfDeformedNodes': arrayOfDeformedNodes})
            else:
                if sent:
                    print('clear')
                    numOfDeformedNodes = len(last_indices)
                    arrayOfDeformedNodes = []               
                    for i in range(numOfDeformedNodes):
                        id = last_indices[i]
                        deformedNodesDict = {
                                'deformedNodeID': str(id),
                                'deformedNodeIntensity': 0.
                            }
                        arrayOfDeformedNodes.append(deformedNodesDict)
                    sent = False
                    #t3
                    timens = time.time_ns()
                    exposed_thing.emit_event('skinDeformedDetection', {'timeStamp': timens, 'numOfDeformedNodes': numOfDeformedNodes, 'arrayOfDeformedNodes': arrayOfDeformedNodes})
                # else:
                #     continue

                # exposed_thing.emit_event('contactAreaInformed', {'timeStamp': time.time_ns(), 'numOfDeformedNodes': numOfDeformedNodes, 'arrayOfDeformedNodes': arrayOfDeformedNodes})

    event_loop = asyncio.get_event_loop()
    event_loop.create_task(send_skin_state(exposed_thing))

@tornado.gen.coroutine
def main():
    LOGGER.info('Creating WebSocket server on: {}'.format(WEBSOCKET_PORT))
    ws_server = WebsocketServer(port=WEBSOCKET_PORT)

    LOGGER.info('Creating HTTP server on: {}'.format(HTTP_PORT))
    http_server = HTTPServer(port=HTTP_PORT)

    LOGGER.info('Creating servient with TD catalogue on: {}'.format(CATALOGUE_PORT))
    servient = Servient(catalogue_port=CATALOGUE_PORT)
    servient.add_server(ws_server)
    servient.add_server(http_server)

    LOGGER.info('Starting servient')
    wot = yield servient.start()

    LOGGER.info('Exposing and configuring Thing')

    # Produce the Thing from Thing Description
    exposed_thing = wot.produce(json.dumps(TD))

    # exposed_thing.set_property_read_handler("skinShape", read_shape_hanlder)
  
    yield exposed_thing.properties['skinMaterial'].write({
        'materialType': 'dragonskin',
        'materialShoreHardness': 'Shore D'
    })

    yield exposed_thing.properties['skinShape'].write({
        'name': 'barrel',
        'baseRadiusDimension': 60,
        'middleRadiusDimension': 80,
        'heightDimension': 260
    })

    yield exposed_thing.properties['sensorCoordinateFrame'].write({
        'originLocation': 'centerOfBottomCricle',
        'axisOrientation': {'xAsix': 'forward', 'yAsix': 'left','zAsix': 'up'}
    })
    
    init_points, skin_cells = load_resources(vtk_file='./resources/skin.vtk')
    # create dictionary for skinNodes dataschema
    skinNodeData = createSkinNodesProperty(init_points)
    yield exposed_thing.properties['skinNodes'].write(skinNodeData)
    # create dictionary for skinCells dataschema
    skinCellData = createSkinCellsProperty(skin_cells)
    yield exposed_thing.properties['skinCells'].write(skinCellData)

    exposed_thing.expose()

    create_touch_detected_task(exposed_thing)
    create_skin_state_update_task(exposed_thing)

    LOGGER.info(f'{TD["title"]} is ready')

if __name__ == '__main__':
    LOGGER.info('Starting loop')
    IOLoop.current().add_callback(main)
    IOLoop.current().start()
