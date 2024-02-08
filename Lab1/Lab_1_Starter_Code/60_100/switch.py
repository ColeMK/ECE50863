#!/usr/bin/env python

"""This is the Switch Starter Code for ECE50863 Lab Project 1
Author: Xin Du
Email: du201@purdue.edu
Last Modified Date: December 9th, 2021
"""

import sys
import time
import socket
import threading
from datetime import date, datetime
import sched
import ast

# Please do not modify the name of the log file, otherwise you will lose points because the grader won't be able to find your log file
LOG_FILE = "switch#.log" # The log file for switches are switch#.log, where # is the id of that switch (i.e. switch0.log, switch1.log). The code for replacing # with a real number has been given to you in the main function.

# Those are logging functions to help you follow the correct logging standard

# "Register Request" Format is below:
#
# Timestamp
# Register Request Sent

def register_request_sent():
    log = []
    log.append(str(datetime.time(datetime.now())) + "\n")
    log.append(f"Register Request Sent\n")
    write_to_log(log)

# "Register Response" Format is below:
#
# Timestamp
# Register Response Received

def register_response_received():
    log = []
    log.append(str(datetime.time(datetime.now())) + "\n")
    log.append(f"Register Response received\n")
    write_to_log(log) 

# For the parameter "routing_table", it should be a list of lists in the form of [[...], [...], ...]. 
# Within each list in the outermost list, the first element is <Switch ID>. The second is <Dest ID>, and the third is <Next Hop>.
# "Routing Update" Format is below:
#
# Timestamp
# Routing Update 
# <Switch ID>,<Dest ID>:<Next Hop>
# ...
# ...
# Routing Complete
# 
# You should also include all of the Self routes in your routing_table argument -- e.g.,  Switch (ID = 4) should include the following entry: 		
# 4,4:4

def routing_table_update(routing_table):
    log = []
    log.append(str(datetime.time(datetime.now())) + "\n")
    log.append("Routing Update\n")
    for row in routing_table:
        log.append(f"{row[0]},{row[1]}:{row[2]}\n")
    log.append("Routing Complete\n")
    write_to_log(log)

# "Unresponsive/Dead Neighbor Detected" Format is below:
#
# Timestamp
# Neighbor Dead <Neighbor ID>

def neighbor_dead(switch_id):
    log = []
    log.append(str(datetime.time(datetime.now())) + "\n")
    log.append(f"Neighbor Dead {switch_id}\n")
    write_to_log(log) 

# "Unresponsive/Dead Neighbor comes back online" Format is below:
#
# Timestamp
# Neighbor Alive <Neighbor ID>

def neighbor_alive(switch_id):
    log = []
    log.append(str(datetime.time(datetime.now())) + "\n")
    log.append(f"Neighbor Alive {switch_id}\n")
    write_to_log(log) 

def write_to_log(log):
    with open(LOG_FILE, 'a+') as log_file:
        log_file.write("\n\n")
        # Write to log
        log_file.writelines(log)

def sendTo(send_socket, msg,addr):
    print(f"Sending message to {addr}")
    msg = msg.encode('utf-8')
    # msg_len = len(msg)
    # send_length = str(msg_len).encode('utf-8')
    # send_length += b' ' * (HEADER - len(send_length))
    # client_socket.sendto(send_length,addr)
    send_socket.sendto(msg, addr)

def topoUpdate(socket,id, liveness,serverAddr):
    msgList = [liveness,id] #liveness and who its from
    msg = f"[LIVENESS]|{msgList}"
    sendTo(socket, msg, serverAddr) #Should send topology update to server

#Run every 2 Seconds
def keepAliveTransmission(socket,id, liveness, neighbors, server): #this functions sends out signals to other terminals and to controler
    for k,v in liveness.items():
        if v[0] == "Alive":
            #Send Alive message
            msg = f"[ALIVE]|{id}"
            neighborAddr = neighbors[k][1]
            sendTo(socket, msg, neighborAddr)
    #Send liveness update to controller
    topoUpdate(socket, id, liveness, server)


#This will consistently be running
def checkAlive(socket, liveness,neighbors, my_id, server_addr):
    while True:
        (data, addr) = socket.recvfrom(1024)
        msg = data.decode("utf-8") # this should be the message with liveness
        msg = msg.split('|')

        print(f"Message in Check alive {msg}")
        if msg[0] == "[ALIVE]":
            print(f"check alive")
            id = ast.literal_eval(msg[1])
            print(f"ID: {id}")

            if liveness[id][0] == "Dead":
                neighbors[id] = [1,addr]
                topoUpdate(socket, my_id, liveness, server_addr)
                neighbor_alive(id)

            liveness[id] = ["Alive", time.time()]
        if msg[0] == "[REGISTER RESPONSE]":
            neighbors = ast.literal_eval(msg[1])
            neighbors = neighbors['neighbors']
        if msg[0] == '[Topology Update]':
            topoList = ast.literal_eval(msg[1])
            print(topoList)
            routing_table_update(topoList)
            # may remove
            for k,v in neighbors.items():
                # if v[0] == 1:
                #     liveness[k] = ["Alive",time.time()]
                # else:
                #     liveness[k] = ["Dead",time.time()]
                liveness[k] = ["Alive",time.time()]
            
    return liveness, neighbors #may not need to need to find out if it passes by address
            
#Consistently Run
def checkDead(client_socket, liveness, timeout, my_id, server_addr):
    deadIDs = {}
    while True:
        oldDeadIDs = deadIDs.copy()
        for key, value in liveness.items():
            # if value[0] == "Dead" and key not in deadIDs:
            #     deadIDs[key] = 0   
            if value[0] == "Alive" and key in deadIDs:
                deadIDs.pop(key)   
        for k,v in liveness.items():
            timDelta = time.time()-v[1]
            #print(f"liveness: {liveness}")
            #print(deadIDs)
            if (timDelta >= timeout):
                #print("here2")
                if k not in deadIDs:
                    deadIDs[k] = 0  
                if (deadIDs[k] == 0):
                    deadIDs[k] = 1
                    liveness[k] = ["Dead", v[1]]
                    neighbor_dead(k)
                    topoUpdate(client_socket, my_id, liveness, server_addr)
                    print(f"{k} Dead {timDelta}")
                    
    
    return liveness #may not need to need to find out if it passes by address

#These will handle threading functions and should need to be left after called

def periodicOperationHandler(socket,id, liveness, neighbors, server, k):
    while True:
        time.sleep(k) #Will make sure it runs every 2 seconds so long as k is 2
        keepAliveTransmission(socket,id,liveness,neighbors, server)

def persistentOperationHandler(socket, liveness, neighbors, id, server, timeout):
    #may add sleep if running too much
    while True:
        liveness, neighbors = checkAlive(socket, liveness, neighbors, id, server)
        liveness = checkDead(liveness, timeout, id, server)
    


            
def main():

    global LOG_FILE
    K = 2
    TIMEOUT = 3*K
    #Check for number of arguments and exit if host/port not provided
    num_args = len(sys.argv)
    if num_args < 4:
        print ("switch.py <Id_self> <Controller hostname> <Controller Port>\n")
        sys.exit(1)

    my_id = int(sys.argv[1])
    LOG_FILE = 'switch' + str(my_id) + ".log" 

    # Write your code below or elsewhere in this file
    PORT = int(sys.argv[3])
    HOSTNAME = str(sys.argv[2])
    ID = int(sys.argv[1])
    HEADER =  64
    DISCONNECTION = "!DISCONNECT" #if send to server will disconnect switch

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (HOSTNAME,PORT)
    print(f"Before sending data the socket address is {client_socket.getsockname()}")
    #client_socket.connect(addr)

    registerRequestMsg = f"[REGISTERREQUEST]|{ID}"
    sendTo(client_socket, registerRequestMsg,addr)
    register_request_sent()
    (data, server_addr) = client_socket.recvfrom(1024)
    msg = data.decode("utf-8") # this should be the message with neighbors and liveness and addresses
    print(msg) # see format

    #Should be register Response
    msgList = msg.split('|')
    neighbors = ast.literal_eval(msgList[1]) #This brings my dictionary in
    print(neighbors)

    register_response_received()

    #-------------Recieve Topology--------------------#

    (data, server_addr) = client_socket.recvfrom(1024)
    topoMessage = data.decode("utf-8") 
    
    topoList = topoMessage.split('|')
    print(topoList)
    if topoList[0] == '[Topology Update]':
        topoList = ast.literal_eval(topoList[1])
        print(topoList)
        routing_table_update(topoList)
    #-------------Keep Alive Signals------------------#
    # threading.Timer(K, periodicOperationHandler).start()

    scheduler = sched.scheduler()
    #Currently Assuming all start live
    liveness = {}
    neighbors = neighbors['neighbors']
    print(neighbors)
    for k,v in neighbors.items():
        liveness[k] = ["Alive",time.time()] #Alive and time since heard from
    
    #Use this way if other doesnt work
    # while True:
    #     (data, server_addr) = client_socket.recvfrom(1024)
    #     msg = data.decode("utf-8") # this should be the message with liveness
    #     msg = msg.split('|')
    #     if msg[0] == "[ALIVE]":
    

    #Will need to make sure its being accessed with k[1]
    periodicThread = threading.Thread(target=periodicOperationHandler, args=(client_socket,ID,liveness,neighbors,server_addr,K))
    persistentThread = threading.Thread(target=persistentOperationHandler, args = (client_socket,liveness,neighbors, ID,server_addr,TIMEOUT))
    checkAliveThread = threading.Thread(target=checkAlive, args=(client_socket, liveness,neighbors, ID, server_addr))
    checkDeadThread = threading.Thread(target=checkDead, args=(client_socket, liveness, TIMEOUT, ID, server_addr))
   
    periodicThread.start()
    #persistentThread.start()
    checkAliveThread.start()
    checkDeadThread.start()




if __name__ == "__main__":
    main()