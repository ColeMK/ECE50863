#!/usr/bin/env python3
from monitor import Monitor
import sys
import time 
import threading
# Config File
import configparser
import zlib
import ast
# def compress_data(data):
#     compressed_data = zlib.compress(data)
#     return compressed_data


# def divide_packets(filename, packet_size): #Creates each packet thing
# 	packetList = []
# 	seekTrack = 0
# 	with open(filename, 'rb') as f:
# 		data = f.read()
# 		compressed_data = compress_data(data)
# 	for i in range(0, len(compressed_data), packet_size):
# 		if i + packet_size > len(compressed_data):
# 			last_packet = compressed_data[(len(compressed_data)//packet_size)*packet_size:]
# 			packetList.append(last_packet)
# 			break
# 		packet = compressed_data[i:i+packet_size]
# 		#print(type(packet))
# 		packetList.append(packet)
	
# 	return packetList, len(packetList)
#backup
def divide_packets(filename, packet_size): #Creates each packet thing
	packetList = []
	seekTrack = 0
	with open(filename, 'rb') as f:
		while True:
			packet = f.read(packet_size)
			#print(len(packet))
			if not packet:
				break
			# seekTrack+=packet_size
			# f.seek(seekTrack)
			packetList.append(packet)
	# print(packetList[0]+packetList[1])	
	return packetList, len(packetList)

class SendThread(threading.Thread):
	#New class for send thread so that it can be stopped and restarted
	#Similar idea to an event thread but wiht more control
    def __init__(self,  *args, **kwargs):
        super(SendThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

class send_packets:
	def __init__(self, packet_list, timeout, send_monitor, receiver_id, window_size, max_packet_size):
		self.packet_list = packet_list
		self.packets_left = len(packet_list)
		self.timeout = timeout
		self.current_packet = None
		self.send_monitor = send_monitor
		self.reciever_id = receiver_id
		self.start_packet_id = 0
		self.curr_packet_id = 0
		self.time_dict = {i: -1 for i in range(len(packet_list))} # added this might change
		self.received_ack = []
		self.to_do = 0
		self.sent_num = 0
		self.done = 0
		#self.resend = -1
		self.window_size = window_size
		self.max_packet_size = max_packet_size
		self.to_send = self.window_size
		self.stop_thread = 0
		self.timer_thread = None
		self.send_thread = None
		self.receive_thread = None
		self.multiple = 0
		self.prev_window_size = window_size
	# def timer_init(self):
	# 	self.timer_thread = threading.Timer(self.timeout, self.timeout_func)
	# 	self.timer_thread.start()

	# def timeout_func(self):
	# 	#print("timeout Func")
	# 	while True:
	# 		#time.sleep(.01)
	# 		#print("waiting timeout")
	# 		#running keys maybe
	# 		#checkDict = self.time_dict.copy()
	# 		if self.done == 1:
	# 			break
	# 		if len(self.time_dict) != 0:
	# 			#print(self.time_dict)
	# 			for k,v in self.time_dict.items():
	# 				# self.time_dict = {key: -1 for key in self.time_dict}
	# 				if (time.time()-v > self.timeout and v != -1) or self.resend > -1:
	# 					if self.resend > -1:
	# 						k = self.resend
	# 						self.resend = -1
	# 					#print(f"length of dict = {len(self.time_dict)}")
	# 					self.time_dict = {key: -1 for key in self.time_dict}
	# 					#print(f"length of dict = {len(self.time_dict)}")
	# 					#print(f'timeout {k}')
	# 					#stop the send thread
	# 					self.stop_thread = 1
	# 					self.send_thread.stop()
	# 					#self.send_thread.join()
	# 					self.to_send = self.window_size
	# 					self.to_do = k-1
	# 					while self.send_thread.is_alive():
	# 						time.sleep(.001)
	# 					self.send_thread = SendThread(target=self.send_func)
	# 					self.send_thread.start()

	# 					break
	# 					#restart send thread
	# 	#print("exiting timeout")
	# def send_func(self):
	# 	self.stop_thread = 0
	# 	#print("send Func")
	# 	self.sent_num = self.to_do #this reinitializes where we left off after timeout due to drop or out of order
	# 	while True:
	# 		if self.stop_thread == 1 or self.done == 1:
	# 			#print("send func stopping")
	# 			break
	# 		while self.to_send > 0 and self.sent_num != len(self.packet_list):
				
	# 			if self.stop_thread == 1 or self.done == 1:
	# 				#print("send func stopping")
	# 				break
	# 			#print(f"Stop thread is: {self.stop_thread}")
	# 			#packet sending info
	# 			# on each packet sent -1
	# 			#print(f"Sent_num: {self.sent_num},{self.to_send}")
	# 			self.current_packet = self.packet_list[self.sent_num]
	# 			self.curr_packet_id = self.sent_num
	# 			msg = (f"{self.curr_packet_id}|").encode('utf-8') + self.current_packet
	# 			#print(len(msg))
	# 			self.send_monitor.send(self.reciever_id,msg)
	# 			self.time_dict[self.curr_packet_id] = time.time()
	# 			self.sent_num += 1
	# 			self.to_send -= 1
	# 			time.sleep(.001)
	# 	#print("leaving send")

	# def receiving_func(self):
	# 	#print("recieving Func")
	# 	new = 0
	# 	curr_ack = None
	# 	prev_ack = 1
	# 	while True:
	# 		if curr_ack:
	# 			prev_ack = curr_ack
	# 		#print("waiting recieving")
	# 		addr, data = send_monitor.recv(self.max_packet_size)
	# 		data = data.decode('utf-8')
	# 		data = data.split('|')
	# 		#print(data)
	# 		if data[0] == "END": 
	# 			time.sleep(.1)
	# 			#print(f"Ending: {time.time()}")
	# 			send_monitor.send_end(receiver_id)
	# 			break
	# 		packet_num = int(data[1])
	# 		if packet_num == len(self.packet_list)-1:
	# 			self.done = 1
	# 			addr, data = send_monitor.recv(max_packet_size)
	# 			data = data.decode('utf-8')
	# 			if data == "END": #may possibly change to waiting and then end
	# 				time.sleep(.1)
	# 				#print(f"Ending: {time.time()}")
	# 				send_monitor.send_end(receiver_id)
	# 				break
	# 		if data[0] == "ACK": #cancels timeout on recieve of ACK
	# 			curr_ack = int(packet_num)
	# 			if curr_ack == prev_ack and new == 0:
	# 				self.resend = curr_ack
	# 				new = 1
	# 			else:
	# 				new = 0
	# 			self.time_dict[packet_num] = -1 
	# 			self.to_send += 1
	# 			if packet_num not in self.received_ack:
	# 				self.received_ack.append(packet_num)
			
		#print(f"leaving recieve, done = {self.done}")
	def timeout_func(self):
		#print("timeout")
		window_size = self.prev_window_size
		time_sent_num = self.sent_num - self.prev_window_size
		print("timeout")
		for i in range(window_size):
			cp = self.packet_list[time_sent_num]
			cid = time_sent_num
			print(f"{self.curr_packet_id}:{self.packets_left}:{window_size}")
			msg = (f"{cid}|{window_size}|").encode('utf-8') + cp
			#print(len(msg))
			self.send_monitor.send(self.reciever_id,msg)
			time_sent_num += 1
			self.send_monitor.send(self.reciever_id,msg) #Should resend packet on timeout
		self.timer_init()

	def timer_init(self):
		self.timer_thread = threading.Timer(self.timeout, self.timeout_func)
		self.timer_thread.start()

	def resend(self,data):
		data = ast.literal_eval(data)
		#data = [eval(i) for i in data]
		#print(f"In data: {data}, Multiple {self.multiple}")
		for i in data:
		#	print(i)
			packet_num = self.window_size * self.multiple + i
			resend_packet = self.packet_list[packet_num]
			msg = (f"{packet_num}|-1|").encode('utf-8') + resend_packet
		#	print(f"resend packet num {packet_num}")
			self.send_monitor.send(self.reciever_id,msg)

	def resend_thread_init(self,data):
		self.resend_thread = threading.Thread(target=self.resend, args=[data[2]])
		self.resend_thread.start()
	def send_thread_init(self):
		self.send_thread = threading.Thread(target=self.send_func)
		self.send_thread.start()
	def receiving_func(self):
		dataBuffer = -1
		while True:
			
			addr, data = send_monitor.recv(self.max_packet_size)
			raw_data = data
			data = data.decode('utf-8')
			data = data.split('|',2)
			if raw_data != dataBuffer:
				if data[0] == "E": 
					self.timer_thread.cancel()
					#print(f"Ending: {time.time()}")
					send_monitor.send_end(receiver_id)
					break
				if data[0] == "N":
					self.timer_thread.cancel()
					self.multiple = int(data[1])
					self.resend_thread_init(data)
					#self.resend(data[2])
				if data[0] == "C":
					self.timer_thread.cancel()
					self.send_thread_init()
					#self.send_func()
			dataBuffer = raw_data


	def send_func(self): 
		window_size = self.window_size
		if self.packets_left < window_size:
			window_size = self.packets_left
		for i in range(window_size):
			self.current_packet = self.packet_list[self.sent_num]
			self.curr_packet_id = self.sent_num
			#print(f"{self.curr_packet_id}:{self.packets_left}:{window_size}")
			msg = (f"{self.curr_packet_id}|{window_size}|").encode('utf-8') + self.current_packet
			#print(len(msg))
			self.send_monitor.send(self.reciever_id,msg)
			self.sent_num += 1
		self.timer_init()
		self.packets_left -= window_size
		self.prev_window_size = window_size

	def start_up(self): #starts the trheads
		for i,packet in enumerate(self.packet_list):
			self.time_dict[i] = -1
		#self.timer_thread = threading.Thread(target=self.timeout_func)
		#self.send_thread = SendThread(target=self.send_func)
		self.receive_thread = threading.Thread(target=self.receiving_func)
		
		self.receive_thread.start()
		self.send_func()


#Need to figure out how to end it if receiver drops the ending packet after last ack just give it a bit

if __name__ == '__main__':
	#print("Sender starting up!")
	config_path = sys.argv[1]

	# Initialize sender monitor
	send_monitor = Monitor(config_path, 'sender')
	
	# Parse config file
	cfg = configparser.RawConfigParser(allow_no_value=True)
	cfg.read(config_path)
	receiver_id = int(cfg.get('receiver', 'id'))
	file_to_send = cfg.get('nodes', 'file_to_send')
	max_packet_size = int(cfg.get('network', 'MAX_PACKET_SIZE'))
	prop_delay = float(cfg.get('network','PROP_DELAY'))
	window_size = int(cfg.get('sender', 'window_size'))

	# if window_size < 10:
	# 	slack = .2
	# else:
	# 	slack = min(window_size * .02, 1)
	slack = 2
	#look at time to send though
	RTT = 2*prop_delay + slack #wait time
	packetList,totpackets = divide_packets(file_to_send, max_packet_size-12) # each packet should be a binary as I read binary got rid of -8
	#print(f"total packets: {totpackets}")

	# Exchange messages!
	packet_sender = send_packets(packetList, RTT, send_monitor, receiver_id, window_size, max_packet_size)
	packet_sender.start_up() #should run through the packet list i segmented
	#print(packet_sender.decodedList[0])
	# print('Sender: Sending "Hello, World!" to receiver.')
	# send_monitor.send(receiver_id, b'Hello, World!')
	# addr, data = send_monitor.recv(max_packet_size)
	# print(f'Sender: Got response from id {addr}: {data}')

	# Exit! Make sure the receiver ends before the sender. send_end will stop the emulator.
	#print('end')
