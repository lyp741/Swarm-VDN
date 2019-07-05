import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.websocket

import os
import argparse
import msgpack
import io
from PIL import Image
from PIL import ImageOps
import threading
import numpy as np
import time
import datetime
import sys
import pickle as pickle
from agent import Agent
# import cupy.cuda.runtime as rt

parser = argparse.ArgumentParser(description='ml-agent-for-unity')
parser.add_argument('--port', '-p', default='8765', type=int,
                    help='websocket port')
parser.add_argument('--ip', '-i', default='127.0.0.1',
                    help='server ip')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--log-file', '-l', default='../../log/model_test/', type=str,
                    help='reward log file name')
parser.add_argument('--agent-count', '-c', default=1, type=int,
                    help='number of agent')
parser.add_argument('--mode-distribute', '-d', default=False, type=bool,
                    help='mode distribute')
parser.add_argument('--mode-evaluate', '-e', default=False, type=bool,
                    help='mode evaluate no learning')
parser.add_argument('--model', '-m', default='None', type=str,
                    help='model')
args = parser.parse_args()

from tornado.options import define, options
define("port", default=8000, help="run on the given port", type=int)

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        greeting = self.get_argument('greeting', 'Hello')
        # a = 1/0
        self.write(greeting + ', friendly user!')

class StatusHandler(tornado.websocket.WebSocketHandler):
    agent = Agent()
    agent_initialized = False
    cycle_counter = 1
    rgb_image_count = 1
    depth_image_count = 0
    depth_image_dim = 0
    ir_count = 1
    ground_count = 0
    compass_count = 1
    target_count = 1

    if args.mode_distribute:
        thread_event = threading.Event()
    
    
    def open(self):
        print("open")

    def on_close(self):
        print("close")

    def on_message(self, message):
        # print("received message")
        self.received_message(message)
        
    def callback(self, count):
        self.write_message('{"inventoryCount":"%d"}' % count)

    def send_action(self, action):
        dat = msgpack.packb({"command": "".join(map(str, action))})
        self.write_message(dat, binary=True)

    def received_message(self, m):
        payload = m
        dat = msgpack.unpackb(payload,  encoding='utf-8')

        image = []
        depth = []
        agent_count = len(dat['image'])
        
        for i in range(agent_count):
            image.append(Image.open(io.BytesIO(bytearray(dat['image'][i]))))
            if (self.depth_image_count == 1):
                depth_dim = len(dat['depth'][0])
                temp = (Image.open(io.BytesIO(bytearray(dat['depth'][i]))))
                depth.append(np.array(ImageOps.grayscale(temp)).reshape(self.depth_image_dim))

        if(self.ir_count == 1):
            ir = dat['ir']
            ir_dim = len(ir[0])
        else:
            ir = []
            ir_dim = 0

        if(self.ground_count == 1):
            ground = dat['ground']
            ground_dim = len(ground[0])
        else:
            ground = []
            ground_dim = 0

        if (self.compass_count == 1):
            compass = dat['compass']
            compass_dim = len(compass[0])
        else:
            compass = []
            compass_dim = 0
            
        if(self.target_count == 1):
            target = dat['target']
            target_dim = len(target[0])
        else:
            target = []
            target_dim = 0
        
        observation = {"image": image, "depth":depth, "ir":ir, "ground":ground, "compass":compass, "target":target}
        reward = np.array(dat['reward'], dtype=np.float32)
        
        end_episode = np.array(dat['endEpisode'], dtype=np.bool)

        actions = self.agent.get_action(observation, 0.8, end_episode)
        if args.model == 'None':
            self.agent.store_experience(observation, actions, reward, end_episode)
            self.agent.learn()
        print(actions, reward)
        self.send_action(actions)

class Application(tornado.web.Application):
    def __init__(self):
        

        handlers = [

            (r'/', IndexHandler),
            (r'/ws', StatusHandler),
        ]

        settings = {
            'template_path': 'templates',
            'static_path': 'static'
        }

        tornado.web.Application.__init__(self, handlers, **settings)
print("???")
if __name__ == '__main__':
    tornado.options.parse_command_line()
    app = Application()
    server = tornado.httpserver.HTTPServer(app)
    server.listen(8765)
    print("start")
    tornado.ioloop.IOLoop.instance().start()
