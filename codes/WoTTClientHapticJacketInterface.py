"""
Web of Tactile Thing (WoTT client) that interfaces with the python API Haptic Jacket
Dependencies: 
- wotpy
- HapticAPI
"""

import asyncio
import logging
from wotpy.wot.servient import Servient
from wotpy.wot.wot import WoT
import numpy as np
import time
import argparse
import utils

logging.basicConfig()
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
parser = argparse.ArgumentParser(description="Url of server")
parser.add_argument("url", type=str)


def points_extraction(init_points_info):
    init_points = list()
    for i in range(init_points_info['numOfNodes']):
        x_init = init_points_info['arrayOfNodes'][i]['nodeLocation']['x']
        y_init = init_points_info['arrayOfNodes'][i]['nodeLocation']['y']
        z_init = init_points_info['arrayOfNodes'][i]['nodeLocation']['z']
        coordinate = [x_init, y_init, z_init]
        init_points.append(coordinate)
    return np.array(init_points)


def cells_extraction(init_cells_info):
    init_cells = list()
    for i in range(init_cells_info['numOfCells']):
        point_1 = int(init_cells_info['arrayOfCells'][i]['connectedNodes'][0])
        point_2 = int(init_cells_info['arrayOfCells'][i]['connectedNodes'][1])
        point_3 = int(init_cells_info['arrayOfCells'][i]['connectedNodes'][2])
        points = [point_1, point_2, point_3]

        init_cells.append(points)
    return np.array(init_cells)

def time_eval(item):
    t = (time.time_ns() - int(item.data))*10**-9
    print(t)


class TimeEval(object):
    def __init__(self):
        self.time_data = list()
    def time_eval(self, item):
        # print(int(item.data))
        # print()
        t_now = time.time_ns()
        t = (t_now - int(item.data)) * 10 ** -9- 0.000140
        if len(self.time_data) < 10:
            self.time_data.append(t)
            print(t)
        elif len(self.time_data) == 10:
            self.time_data = np.array(self.time_data)
            mean_time = np.mean(self.time_data)
            std_time = np.std(self.time_data)
            print('mean of time: ',mean_time)
            print('std of time: ',std_time)

async def main():

    t_start = time.time_ns()
    wot = WoT(servient=Servient())
    url_server = parser.parse_args()
    consumed_thing = await wot.consume_from_url(url_server.url)
    LOGGER.info('Consumed Thing: {}'.format(consumed_thing))
    init_points_info = await consumed_thing.read_property('skinNodes')
    init_points = points_extraction(init_points_info)
    init_cells_info = await consumed_thing.read_property('skinCells')
    init_cells = cells_extraction(init_cells_info)
    mat = await consumed_thing.read_property('skinMaterial')
    shape = await consumed_thing.read_property('skinShape')
    contact_info = await consumed_thing.read_property('contactInformation')
    sensor_coor = await consumed_thing.read_property('sensorCoordinateFrame')
    tacSense = utils.SensingCells(num_angular_cells=2, num_height_cells=2, init_points=init_points)
    consumed_thing.events['skinDeformedDetection'].subscribe(
        on_next=tacSense.update,
        on_completed=LOGGER.info('Subscribed for an event: skinDeformedDetection'),
        on_error=lambda error: LOGGER.info(f'Error for an event skinDeformedDetection: {error}'),
    )


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()

