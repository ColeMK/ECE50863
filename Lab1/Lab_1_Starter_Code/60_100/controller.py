#!/usr/bin/env python

"""This is the Controller Starter Code for ECE50863 Lab Project 1
Author: Xin Du
Email: du201@purdue.edu
Last Modified Date: December 9th, 2021
"""

import sys
import socket
import threading
from datetime import date, datetime
import heapq
import json
import time
import ast

# Please do not modify the name of the log file, otherwise you will lose points because the grader won't be able to find your log file
LOG_FILE = "Controller.log"

# Those are logging functions to help you follow the correct logging standard

# "Register Request" Format is below:
#
# Timestamp
# Register Request <Switch-ID>

def register_request_received(switch_id):
    log = []
    log.append(str(datetime.time(datetime.now())) + "\n")
    log.append(f"Register Request {switch_id}\n")
    write_to_log(log)

# "Register Responses" Format is below (for every switch):
#
# Timestamp
# Register Response <Switch-ID>

def register_response_sent(switch_id):
    log = []
    log.append(str(datetime.time(datetime.now())) + "\n")
    log.append(f"Register Response {switch_id}\n")
    write_to_log(log) 

# For the parameter "routing_table", it should be a list of lists in the form of [[...], [...], ...]. 
# Within each list in the outermost list, the first element is <Switch ID>. The second is <Dest ID>, and the third is <Next Hop>, and the fourth is <Shortest distance>
# "Routing Update" Format is below:
#
# Timestamp
# Routing Update 
# <Switch ID>,<Dest ID>:<Next Hop>,<Shortest distance>
# ...
# ...
# Routing Complete
#
# You should also include all of the Self routes in your routing_table argument -- e.g.,  Switch (ID = 4) should include the following entry: 		
# 4,4:4,0
# 0 indicates ‘zero‘ distance
#
# For switches that can’t be reached, the next hop and shortest distance should be ‘-1’ and ‘9999’ respectively. (9999 means infinite distance so that that switch can’t be reached)
#  E.g, If switch=4 cannot reach switch=5, the following should be printed
#  4,5:-1,9999
#
# For any switch that has been killed, do not include the routes that are going out from that switch. 
# One example can be found in the sample log in starter code. 
# After switch 1 is killed, the routing update from the controller does not have routes from switch 1 to other switches.

def routing_table_update(routing_table):
    log = []
    log.append(str(datetime.time(datetime.now())) + "\n")
    log.append("Routing Update\n")
    for row in routing_table:
        log.append(f"{row[0]},{row[1]}:{row[2]},{row[3]}\n")
    log.append("Routing Complete\n")
    write_to_log(log)

# "Topology Update: Link Dead" Format is below: (Note: We do not require you to print out Link Alive log in this project)
#
#  Timestamp
#  Link Dead <Switch ID 1>,<Switch ID 2>

def topology_update_link_dead(switch_id_1, switch_id_2):
    log = []
    log.append(str(datetime.time(datetime.now())) + "\n")
    log.append(f"Link Dead {switch_id_1},{switch_id_2}\n")
    write_to_log(log) 

# "Topology Update: Switch Dead" Format is below:
#
#  Timestamp
#  Switch Dead <Switch ID>

def topology_update_switch_dead(switch_id):
    log = []
    log.append(str(datetime.time(datetime.now())) + "\n")
    log.append(f"Switch Dead {switch_id}\n")
    write_to_log(log) 

# "Topology Update: Switch Alive" Format is below:
#
#  Timestamp
#  Switch Alive <Switch ID>

def topology_update_switch_alive(switch_id):
    log = []
    log.append(str(datetime.time(datetime.now())) + "\n")
    log.append(f"Switch Alive {switch_id}\n")
    write_to_log(log) 

def write_to_log(log):
    with open(LOG_FILE, 'a+') as log_file:
        log_file.write("\n\n")
        # Write to log
        log_file.writelines(log)

def config_handler(configFile):
    with open(configFile, "r") as file:
        data = file.read()
        dataList = data.split('\n')
        if dataList[-1] == '':
            dataList.pop(-1)
        totSwitch = int(dataList[0])
        #graphDict = dict.fromkeys(range(totSwitch), dict())
        graphDict = {}
        for x in range(totSwitch):
            graphDict[x] = {}
        for i,value in enumerate(dataList):
            if i != 0:
                valueList = value.split(' ')
                valueList = [int(item) for item in valueList]
                for j,data in enumerate(valueList):
                    if j !=2:
                        if j == 0:
                            graphDict[valueList[0]][valueList[1]] = valueList[2]
                            print(valueList[0])
                        elif j == 1:
                            graphDict[valueList[1]][valueList[0]] = valueList[2]
    return totSwitch, graphDict

def sendTo(send_socket, msg,addr):
    print(f"Sending message: {msg} to {addr}")
    msg = msg.encode('utf-8')
    send_socket.sendto(msg, addr)

def dijkstra(graphDict, id, neighborDict, distances, paths, liveness):

    distances[id] = 0
    paths[id] = [id] #Shortest path to itself is itself
    pq = [(0,id)]
    for liveID,value in neighborDict[id].items():
        if liveness[liveID][0] == 'Dead':
            neighborDict[id][liveID] = [0, neighborDict[id][liveID][1]]
    #This is meant to be for link fail
    tempGraphDict = graphDict.copy()
    #print(f"Tempgraph before {tempGraphDict}")
    for k,v in neighborDict[id].items():
        if v[0] == 0:
            if k in tempGraphDict[id].keys():
                if tempGraphDict[id][k]:
                    tempGraphDict[id].pop(k)
            #tempGraphDict[id].pop(key)
    print(f"For id {id} Tempgraph after {tempGraphDict} and neightbor dict is {neighborDict}")
    ######################
    # for key, item in tempGraphDict[id].items():
    #    if  key not in tempGraphDict.keys():
    #        tempGraphDict[id].pop(key)
    while pq:
        cd, cn = heapq.heappop(pq) #current distance and current node
        jumpFlag = 0
        if cd <= distances[cn]: # may need to change
            #print(f"cn is {cn} and temp graph dict is {tempGraphDict}")
            if cn in tempGraphDict:
                for neighbor, distance in tempGraphDict[cn].items():
                    newDistance = cd + distance
                    if newDistance < distances[neighbor]:
                        distances[neighbor] = newDistance
                        paths[neighbor] = paths[cn] + [neighbor]
                        heapq.heappush(pq, (newDistance,neighbor))
    
    return distances,paths

def formatRoutingTable(routing):
    formattedRoute = []
    for k,v in routing.items():
        for i,j in v.items():
            formattedRoute.append([k,i,j[0],j[1]])
    return formattedRoute

def generateRoutingTable(graphDict, neighborDict, liveness):
    completeRouting = {}

    #Check our liveness if dead in liveness it should also be dead in neighbor dict already
    tempDict = graphDict.copy()
    print(liveness) ##########################
    for liveID,value in liveness.items():
        if value[0] == 'Dead':
            tempDict.pop(liveID)
    print(f"tempdict is {tempDict}")
    # for liveID,value in tempDict.items():

    #print(tempDict) ############################
    for i in tempDict:
        completeRouting[i] = {}
    for id in tempDict.keys():
        distances = {}
        paths = {}
        for k in graphDict:
            distances[k] = 9999
            paths[k] = []
        routingtable,shortestpaths = dijkstra(tempDict, id, neighborDict,distances, paths, liveness)
        for k,v in shortestpaths.items():
            if len(shortestpaths[k]) == 1: #format is key1 (start), key2 (destination), [next hop, tot distance]
                completeRouting[id][k] = [shortestpaths[k][0], routingtable[k]] 
            elif len(shortestpaths[k]) > 1:
                completeRouting[id][k] = [shortestpaths[k][1], routingtable[k]]
    for k,v in graphDict.items():
        print(f"complete routing is {completeRouting}")
        for key,value in completeRouting.items():
            if k not in completeRouting[key]:
                completeRouting[key][k] = [-1,9999]
    #Put a loop to check through liveness and if a switch id is dead then give it a [-1, 9999]
    return completeRouting

def controllerTopologyUpdate(graphDict,idDict,server_socket, neighborDict, liveness):
    completeRouting = generateRoutingTable(graphDict, neighborDict, liveness)
    controllerRouting = formatRoutingTable(completeRouting)
    routing_table_update(controllerRouting)
    for id,values in idDict.items():
        routingList = []
        for x in controllerRouting:
            if x[0] == id:
                routingList.append(x)
        print("here")
        routingMessage = f"[Topology Update]|{routingList}"
        sendTo(server_socket, routingMessage, idDict[id])

#########_________________NEED FUNCTION TO CHECK IF DEAD_________#

def checkDead(liveness,timeout,graphDict,idDict,server_socket,neighborDict):
    deadIDs = {}
    while True:
        for key, value in liveness.items():
                # if value[0] == "Dead" and key not in deadIDs:
                #     deadIDs[key] = 0   
                if value[0] == "Alive" and key in deadIDs:
                    deadIDs.pop(key)   
        for k,v in liveness.items():
            timeDelta =  time.time() - v[1]
            if (timeDelta >= timeout):
                if k not in deadIDs:
                    deadIDs[k] = 0  
                if (deadIDs[k] == 0):
                    deadIDs[k] = 1
                    liveness[k] = ["Dead", v[1]]
                    # for key, value in neighborDict.items():
                    #     if k in neighborDict[key].keys():
                    #         neighborDict[key][k] = [0, neighborDict[key][k][1]]
                    #topoUpdate(client_socket, my_id, liveness, server_addr) # may not thread the topo update
                    # threadTopoUpdate = threading.Thread(target=controllerTopologyUpdate, args=(graphDict,idDict,server_socket,neighborDict, liveness)) #might only run topology update on the one id include this again if need to update whenever neighbor is dead
                    # threadTopoUpdate.start()
                    topology_update_switch_dead(k)
                    controllerTopologyUpdate(graphDict,idDict,server_socket,neighborDict, liveness)
                    print(f"{k} Dead {timeDelta}")
                    


def switchReciever(server_socket,idDict,liveness, graphDict, neighborDict):
    while True:
        (data, client_addr) = server_socket.recvfrom(1024)
        msg = data.decode("utf-8")
        msg = msg.split('|')
        topoFlag = 0
        if msg[0] == "[LIVENESS]":
            updateList = ast.literal_eval(msg[1])
            usedID = updateList[1]
            liveness[updateList[1]] = ["Alive", time.time()] #Updates the current id 
            print(updateList[0])
            print(f"neightbor dict in switch is {neighborDict}")
            print(f"Liveness is {liveness}")
            for k,v in updateList[0].items():
                # I cant have it checking liveness in here for link deaths
                if v[0] == "Dead" and liveness[k][0] == "Dead": #this might neede to update neighbordict instead
                    print(neighborDict)
                    if neighborDict[usedID][k][0] == 1:
                        topoFlag = 1 # 1 is new dead
                        print(f"Topoflag set in dead")
                        topology_update_link_dead(updateList[1], k) #new death
                    neighborDict[usedID][k][0] = 0
                elif v[0] == "Alive" and liveness[k][0] == "Alive":
                    if neighborDict[usedID][k][0] == 0:
                        topoFlag = 1
                        print(f"Topoflag set in alive {neighborDict}")
                    neighborDict[usedID][k][0] = 1
                #     if liveness[k][0] == "Alive":
                #         topoFlag  = 1
                #     liveness[k] = ["Dead", time.time()]
                #     #include detections to see if these changed from previous values #######################################################
                # else:
                #     if liveness[k][0] == "Dead":
                #         topoFlag  = 1
                #     liveness[k] = ["Alive",liveness[k][1]]
                if topoFlag == 1: 
                    print("IN SWITHC RECIEVER")
                    controllerTopologyUpdate(graphDict,idDict,server_socket,neighborDict, liveness) #might only run topology update on the one id include this again if need to update whenever neighbor is dead

        if msg[0] == "[REGISTERREQUEST]":
            id = ast.literal_eval(msg[1])
            register_request_received(id) # may not need
            smallNeighborDict = {}
            idDict[int(id)] = client_addr
            neighbors = graphDict[id].keys() ## this gets neighbors
            for x in list(neighbors):
                if liveness[x][0] == "Dead":
                    smallNeighborDict[x] = [0,idDict[x]] # if 1 its alive
                else:
                    smallNeighborDict[x] = [1,idDict[x]]
            neighborDict[id] = smallNeighborDict
            responseDict = {"neighbors" : smallNeighborDict, "address" : idDict[id]}
            responseMessage = f"[Register Response]|{responseDict}"
            sendTo(server_socket, responseMessage, idDict[id])
            register_response_sent(id)
            topology_update_switch_alive(id)
            
            controllerTopologyUpdate(graphDict,idDict,server_socket,neighborDict, liveness)

def persistentOperationHandler(server_socket,liveness):
    while True:
        pass

def main():
    #Check for number of arguments and exit if host/port not provided
    num_args = len(sys.argv)
    if num_args < 3:
        print ("Usage: python controller.py <port> <config file>\n")
        sys.exit(1)
    K = 2
    TIMEOUT = 3*K
    # Write your code below or elsewhere in t his file
    IP_ADDR = '0.0.0.0' # might use socket.gethostbyname(socket.gethostname())
    PORT = int(sys.argv[1])
    HEADER = 64 #may be toos mall
    DISCONNETION = "!DISCONNECT"

    totSwitch, graphDict = config_handler(sys.argv[2])
    print(f"Graph Dict: {graphDict}")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #create udp socket
    server_socket.bind((IP_ADDR, PORT))
    idDict = {}
    neighborDict = {}
    overallNeighborDict = {}
    while True:
        (data, client_addr) = server_socket.recvfrom(1024)
        msg = data.decode("utf-8")
        msg = msg.split('|')
        if msg[0] == "[REGISTERREQUEST]":
            id = ast.literal_eval(msg[1])
            register_request_received(id)
            idDict[int(id)] = client_addr

        #-------------------- Handles the register response ---------------#
        if (len(idDict) == totSwitch) and (len(neighborDict) < 1):
            for key, value in idDict.items():
                neighbors = graphDict[key].keys() ## this gets neighbors
                neighborDict = {}
                for x in list(neighbors):
                    neighborDict[x] = [1,idDict[x]] # if 1 its alive
                overallNeighborDict[key] = neighborDict
                responseDict = {"neighbors" : neighborDict, "address" : idDict[key]}
                responseMessage = f"[Register Response]|{responseDict}"
                sendTo(server_socket, responseMessage, idDict[key])
                register_response_sent(key)
            
            break
            # may want to break out of while true after this
        #------------------------------------------------------------------#
    liveness = {}
    deadList = []
    for k,v in idDict.items():
        if k not in deadList:
            liveness[k] = ["Alive",time.time()]
        else:
            liveness[k] = ["Dead",time.time()]
        #-------------------- Shortest Path Calc --------------------------#
    completeRouting = generateRoutingTable(graphDict,overallNeighborDict, liveness)
    controllerRouting = formatRoutingTable(completeRouting)
    routing_table_update(controllerRouting)
    for id,values in idDict.items():
        routingList = []
        for x in controllerRouting:
            if x[0] == id:
                routingList.append(x)
        print("here")
        routingMessage = f"[Topology Update]|{routingList}"
        sendTo(server_socket, routingMessage, idDict[id])

        #------------------------------------------------------------------#
        #-------------------- Keep Alive Handling -------------------------#

    
    # while True:
    #     (data, client_addr) = server_socket.recvfrom(1024)
    #     msg = data.decode("utf-8") # this should be the message with liveness
    #     msg = msg.split('|')
    #     if msg[0] == "[LIVENESS]":
    #         update = ast.literal_eval(msg[1])

        #------------------------------------------------------------------#
    
    switchRecieverThread = threading.Thread(target=switchReciever, args=(server_socket, idDict,liveness, graphDict, overallNeighborDict))
    checkDeadThread = threading.Thread(target=checkDead, args=(liveness, TIMEOUT, graphDict, idDict, server_socket,overallNeighborDict))

    switchRecieverThread.start()
    checkDeadThread.start()
            



    # def client_handler(conn, addr):
    #     print(f"[NEW CONNECTION] {addr} connected")
    #     connection = True
    #     while connection:
    #         msg_len = conn.recv(HEADER).decode('utf-8')
    #         if msg_len:
    #             msg_len = int(msg_len)
    #             msg = conn.recv(msg_len).decode('utf-8') # reads message of size length given above
    #             if msg == DISCONNETION:
    #                 connection = False
    #             print(f"[{addr}] {msg}")

    #     conn.close() #closes current connection on disconnect

    # def start():
    #     #Listens for connections and handles them
    #     server_socket.listen()
    #     print(f"Server listening on {server_socket}")
    #     while True:
    #         conn, addr = server_socket.accept() #checks for new connection to server
    #         # add threading here # 
    #         thread = threading.Thread(target=client_handler, args=(conn, addr))
    #         thread.start()
    #         print(f"ACTIVE CONNECTIONS {threading.activeCount()-1}") #used for me to know how many connections

if __name__ == "__main__":
    main()