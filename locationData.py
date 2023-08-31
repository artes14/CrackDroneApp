import csv
import numpy as np
from datetime import datetime, timedelta
from datetime import time
import os
from pathlib import Path


class locationData:
    def __init__(self, fileName: str = None):
        self.currentLocationData=[]
        self.x, self.y, self.z, self.v = [],[],[],[]
        self.dt_1=23

        if fileName is not None:
            with open(fileName) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count>1:
                        # print('x:{0}, y:{1}, z:{2}, v:{3}'.format(row[0], row[1], row[2], row[3]))
                        self.currentLocationData.append([row[0], row[1], row[2], row[3]])
                        self.x.append(float(row[0]))
                        self.y.append(float(row[1]))
                        self.z.append(float(row[2]))
                        self.v.append(float(row[3]))
                    line_count += 1
        # get image
        if len(os.listdir('cloudImages')) == 0:
            print(' %s ' % 'No image to speculate...')
        else:
            startTime=datetime.now()
            self.time_dic= {}
            isfirst=True
            for p in Path('cloudImages').glob('*'):
                path = str(p)
                name = os.path.basename(path)
                name = '.'.join(name.split('.')[:-1])
                # date = name.split('_')[0]
                t = name.split('_')[1]
                h, m, s =int(t[0:2]), int(t[2:4]), int(t[4:6])
                Time = datetime.combine(datetime.today(), time(h, m, s))
                if isfirst:
                    startTime=Time
                    self.time_dic[name]=0
                    isfirst=False
                else:
                    dt=int((Time-startTime).total_seconds()*self.dt_1)
                    if dt<line_count-1:
                        self.time_dic[name]=dt
            print(self.time_dic, line_count)


# class logData:
#     def __init__(self, fileName: str = None):
#         self.timestamp = []
#         self.d_forward, self.d_uwb1, self.d_uwb2, self.d_down, self.vel_y = [],[],[],[],[]
#         self.dt=0.01
#         if fileName is not None:
#             with open(fileName) as csv_file:
#                 csv_reader = csv.reader(csv_file, delimiter=',')
#                 line_count = 0
#                 for row in csv_reader:
#                     if line_count % 10 == 0:
#                         # print('x:{0}, y:{1}, z:{2}, v:{3}'.format(row[0], row[1], row[2], row[3]))
#                         self.currentLocationData.append([row[0], row[1], row[2], row[3]])
#                         x, y, z, vel = [row[0], row[1], row[2], row[3]]
#                     line_count += 1
#                 print(self.currentLocationData)



