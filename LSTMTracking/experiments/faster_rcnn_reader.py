# Copyright (c) <2017> <Xi Li>. All Rights Reserved.
 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 

'''
read image features and bbox from faster r-cnn dectection, hlcv project ss17
'''

import numpy as np
import os.path

def read_concatenate_all(GT_path):
    bbox_path = os.path.dirname(GT_path) + "/../Estimated_BBox/"
    fts_path = os.path.dirname(GT_path) + "/../Extracted_Features/"
    info = np.genfromtxt(GT_path, dtype = 'unicode', delimiter=',',max_rows = 1)
    
    bbox_path = bbox_path + info[1] + '-bbox.txt'
    fts_path = fts_path + info[1] + '-fts.txt'
    
    GT_data = np.genfromtxt(GT_path, delimiter=',',skip_header = 1)
    bbox_data = np.genfromtxt(bbox_path, delimiter=',')                
    f = open(fts_path, 'rb')
    fts_data = np.array([], dtype='string')
    for line in f:
        fts_data = np.append(fts_data, line)
          
    return GT_data, bbox_data, fts_data

def extractSingleSample(id, bbox_data, fts_data, num_steps, num_input):
    """
    Get single sample with size [num_steps, num_input]. Since we don't have successive detection results from faster r-cnn
    we use last available data if we have lost current one.
    """
    X_ = []
    frame_id = []
    max_num_item = len(bbox_data) - 1
    index = id
    if (id > max_num_item): #fix the item number, since it might not be the last frame
        index = max_num_item
    splitFtsData = np.asarray((fts_data[index].replace("#", "").split()))
    splitBBoxData = bbox_data[index]
    frame_id = np.append(frame_id, int(splitBBoxData[0]))
    X_ = np.append(X_, splitFtsData)
    X_ = np.append(X_, [splitBBoxData[1], splitBBoxData[2], splitBBoxData[3], splitBBoxData[4]])
    for i in range(1, num_steps):
        if (id + i > max_num_item):
            index = max_num_item
        splitFtsData_new = np.asarray((fts_data[index].replace("#", "").split()))
        splitBBoxData_new = bbox_data[index]
        frame_id_new = int(splitBBoxData_new[0])
        if (frame_id_new > frame_id[-1] + 1):            
            X_ = np.append(X_, splitFtsData)
            X_ = np.append(X_, [splitBBoxData[1], splitBBoxData[2], splitBBoxData[3], splitBBoxData[4]])
        else: #WARNING: we might have same frame id data due to faster rcnn results, we always use highest confidence score one
            X_ = np.append(X_, splitFtsData_new)
            X_ = np.append(X_, [splitBBoxData_new[1], splitBBoxData_new[2], splitBBoxData_new[3], splitBBoxData_new[4]])
            splitFtsData = splitFtsData_new
            splitBBoxData = splitBBoxData_new
        frame_id = np.append(frame_id, frame_id[-1] + 1)
    #print frame_id    
    X_ = np.reshape(X_, [num_steps, num_input])
    
    return X_, frame_id.astype(int)

def extractSingleGTBBox(frame_id, GT_data, num_steps):
    #since we have skipped Header
    Y_ = []
    GT_data_item = []
    for j in range(len(frame_id)):
        idx = int(frame_id[j])
        for i in range(len(GT_data)):
            if (GT_data[i][0] == idx):
                GT_data_item = GT_data[idx]
                break
        Y_ = np.append(Y_, [GT_data_item[2], GT_data_item[3], GT_data_item[4], GT_data_item[5]])
    Y_ =  np.reshape(Y_, [num_steps, 4])
    return Y_

def read_GT_list(path):
    list = np.genfromtxt(path, delimiter=',', dtype='string')
    GT_list = []
    
    for i in range(0, len(list)):
        GT_list = np.append(GT_list, os.path.dirname(path) + "/GT/" + list[i])
        
    return GT_list, len(GT_list)
        