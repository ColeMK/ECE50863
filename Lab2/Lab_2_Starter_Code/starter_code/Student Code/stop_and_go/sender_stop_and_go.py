#!/usr/bin/env python3
from monitor import Monitor
import sys
import time 
import threading
# Config File
import configparser

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
	return packetList


class send_packets:
	def __init__(self, packet_list, timeout, send_monitor, receiver_id, bandwidth):
		self.packet_list = packet_list
		self.timer_thread = None
		self.timeout = timeout
		self.current_packet = None
		self.send_monitor = send_monitor
		self.reciever_id = receiver_id
		self.curr_packet_id = 0
		self.decodedList = []
		#self.sleep_time = 1.0/bandwidth * 3
	def timer_init(self):
		self.timer_thread = threading.Timer(self.timeout, self.timeout_func)
		self.timer_thread.start()

	def timeout_func(self):
		msg = f"{self.curr_packet_id}|".encode('utf-8') + self.current_packet
		self.send_monitor.send(self.reciever_id,msg) #Should resend packet on timeout
		self.timer_init()

	def send_func(self):
		for i,packet in enumerate(packetList):
			self.curr_packet_id = i
			self.current_packet = packet
			# msg = f"{self.curr_packet_id}|{self.current_packet}"
			#print(f"{self.curr_packet_id}")
			# msg = msg.encode('utf-8')
			msg = (f"{self.curr_packet_id}|").encode('utf-8') + self.current_packet
			
			#print(len(msg))
			
			#self.decodedList.append(msg.decode('utf-8'))
			#print(type(msg)		
			self.send_monitor.send(self.reciever_id,msg)
			self.timer_init()
			addr, data = send_monitor.recv(max_packet_size)
			data = data.decode('utf-8')
			if data == "ACK": #cancels timeout on recieve of ACK
				self.timer_thread.cancel()
			time.sleep(.05)			

def end_func(reciever_id):
	send_monitor.send_end(reciever_id)
	sys.exit()

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
	
	bandwidth = float(cfg.get('network', 'LINK_BANDWIDTH'))
	slack = max(1.0/bandwidth * 1000, .04) #look at time to send though
	RTT = 2*prop_delay + slack #wait time
	data_limit = int(2*prop_delay*(bandwidth/max_packet_size)) + 1 # The bandwidth delay product * 2
	packetList = divide_packets(file_to_send, max_packet_size-8) # each packet should be a binary as I read binary


	# Exchange messages!
	packet_sender = send_packets(packetList, RTT, send_monitor, receiver_id, bandwidth)
	packet_sender.send_func() #should run through the packet list i segmented
	#print(packet_sender.decodedList[0])
	# print('Sender: Sending "Hello, World!" to receiver.')
	# send_monitor.send(receiver_id, b'Hello, World!')
	# addr, data = send_monitor.recv(max_packet_size)
	# print(f'Sender: Got response from id {addr}: {data}')
	end_timer = threading.Timer(2, end_func, args=(receiver_id))
	end_timer.start()
	# Exit! Make sure the receiver ends before the sender. send_end will stop the emulator.
	addr, data = send_monitor.recv(max_packet_size)
	data = data.decode('utf-8')
	if data == "END": #cancels timeout on recieve of ACK
		#may need to add something to reciever if this end packet is dropped
		end_timer.cancel()
		send_monitor.send_end(receiver_id)