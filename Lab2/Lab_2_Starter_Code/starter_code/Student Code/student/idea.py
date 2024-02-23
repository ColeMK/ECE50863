self.time_dict = {}
self.received_ack = []
self.to_do = 0
self.sent_num = 0
#sending

self.sent_num = self.to_do #this reinitializes where we left off after timeout due to drop or out of order
while True:
    while self.to_send > 0:
        #packet sending info
        # on each packet sent -1
            self.current_packet = packetList[self.sent_num]
            self.curr_packet_id = self.sent_num
            msg = (f"{self.curr_packet_id}|").encode('utf-8') + self.current_packet
            self.send_monitor.send(self.reciever_id,msg)
            self.time_dict[self.curr_packet_id] = time.time()


            # for i,packet in enumerate(packetList):
            # self.curr_packet_id = i
            # self.current_packet = packet
            # msg = (f"{self.curr_packet_id}|").encode('utf-8') + self.current_packet
            # print(len(msg))
            
            # #self.decodedList.append(msg.decode('utf-8'))
            # #print(type(msg)		
            # self.send_monitor.send(self.reciever_id,msg)
            # self.time_dict[self.curr_packet_id] = time.time()
#recieving

while True:
    addr, data = send_monitor.recv(max_packet_size)
    data = data.decode('utf-8')
    data = data.split('|')
    packet_num = int(data[1])
    if data[0] == "ACK": #cancels timeout on recieve of ACK
        self.time_dict[packet_num] = -1 
        if packet_num not in self.received_ack:
             self.to_send += 1
             
        
    
#Timeout
while True:
     #running keys maybe
     for k,v in self.time_dict.items():
          if time.time() > v + self.timeout:
               #stop the send thread
               self.to_send = window size
               self.to_do = k

               #start again from k

