#!/usr/bin/env python3
from monitor import Monitor
import sys
import time
# Config File
import zlib
import configparser
import ast
import threading

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
# 		packetList.append(packet)
	
# 	return packetList, len(packetList)
def divide_packets(filename, packet_size): #Creates each packet thing
	packetList = []
	with open(filename, 'rb') as f:
		while True:
			packet = f.read(packet_size)
			if not packet:
				break
			packetList.append(packet)
	return packetList,len(packetList)

class packet_reciever:
	def __init__(self,sender_id, tot_packets, write_location, max_packet_size,recv_monitor, window_size):
		self.curr_packet = 0
		self.to_write = b''
		self.window_size = window_size
		self.window = [-1] * self.window_size
		self.tracking_pack = [-1] * self.window_size
		self.ids_recieved = []
		self.nack_timer = 0.4
		self.collected_packets = []
		self.end = 0
		self.multiple = 0
		self.sender_id = sender_id
		self.tot_packets = tot_packets
		self.write_location = write_location
		self.max_packet_size = max_packet_size
		self.recv_monitor = recv_monitor
		self.sender_size = window_size
		self.cum_index = 0
		self.max_packet = 0

		self.nack_timer_live = 0

	def timer_init(self):
		self.timer_thread = threading.Timer(self.nack_timer, self.send_nack)
		self.timer_thread.start()

	def write_to_file(self, buff_window, write_location):
		with open(write_location, 'ab') as f:
			for i,p in enumerate(buff_window):
				#print(i)
				if p!= -1:
					f.write(p)

	def send_nack(self): #might send twice for redundancy
		try:
			self.cum_index = self.window[0:self.sender_size].index(-1) # find first occurance of -1
		except:
			self.cum_index = len(self.window) #this may need to change to be a packet value self.tracking_pack[self.sender_size-1] or -1
		
		self.nack_timer_live = 0
		#print(self.window[0:self.sender_size])
		indexes_of_minus_one = [index for index, value in enumerate(self.window[0:self.sender_size]) if value == -1]
		if len(indexes_of_minus_one) != 0 or self.max_packet != tot_packets - 1:

			msg = (f"ACK|{self.tracking_pack[self.cum_index-1]}|{self.cum_index}|NACK|{indexes_of_minus_one}")#{self.multiple}|{indexes_of_minus_one}") #ack the packet before
			msg = msg.encode('utf-8')
			for i in range(2):
				print("send_nack")
				recv_monitor.send(sender_id, msg)
			self.buff_window = self.window[0:self.sender_size]
			self.write_thread = threading.Thread(target=self.write_to_file, args=(self.buff_window, self.write_location))
			self.write_thread.start()
			self.window = self.window[self.cum_index:] + [-1] * self.cum_index #takes it back to proper size
			self.timer_init()
			self.nack_timer_live = 1

		else: #should end
			with open(self.write_location, 'ab') as f:
				for i,p in enumerate(self.window[0:self.sender_size]):
					#print(i)
					if p!= -1:
						f.write(p)
			recv_monitor.recv_end(self.write_location, self.sender_id)
			#print(f"Ending: {time.time()}")
			self.end = 1
			print("END")
			while(True):
				msg = 'END'
				msg = msg.encode('utf-8')
				recv_monitor.send(sender_id, msg)
				time.sleep(.01)
			

	def send_com(self):
		while True:
			my_slice = self.tracking_pack[0:self.sender_size]
			
			if -1 not in my_slice:
				if self.max_packet < tot_packets-1:
					self.timer_thread.cancel()
					self.nack_timer_live = 0
					print("send com")
					msg = (f"COM|{self.multiple}")
					msg = msg.encode('utf-8')
					for i in range(2):
						recv_monitor.send(sender_id, msg)
					self.multiple +=1
					self.buff_window = self.window[0:self.sender_size]
					#print(self.buff_window)
					with open(write_location, 'ab') as f:
						for i,p in enumerate(self.buff_window):
							#print(i)
							f.write(p)
					self.window = [-1] * self.window_size
					self.tracking_pack = [-1] * self.window_size

				else:
					self.timer_thread.cancel()
					self.nack_timer_live = 0
					
					self.buff_window = self.window[0:self.sender_size]
					#print(self.buff_window)
					with open(write_location, 'ab') as f:
						for i,p in enumerate(self.buff_window):
							#print(i)
							f.write(p)
					self.window = [-1] * self.window_size
					self.tracking_pack = [-1] * self.window_size
					recv_monitor.recv_end(self.write_location, self.sender_id)
					#print(f"Ending: {time.time()}")
					self.end = 1
					print("END")
					while(True):
						msg = 'END'
						msg = msg.encode('utf-8')
						recv_monitor.send(sender_id, msg)
						time.sleep(.01)
						#maybe sys exit
			
			
	
	def start_up(self):
		#self.timer_thread = threading.Timer(self.nack_timer, self.send_nack())
		#self.send_com_thread = threading.Thread(target = self.send_com)
		#self.send_com_thread.start()
		while self.end != 1: # may be -1
			addr, data = recv_monitor.recv(max_packet_size)
			# try:
			# 	data = data.decode('utf-8', errors="ignore")
			# except UnicodeDecodeError as e:
			# 	raise Exception("Error decoding byte string:", e)
			#print(f"Data: {len(data)}")

			data = data.split(b'|',2) #would need to use different split if it was in one of those files
			#print(f"dataa length: {len(data)}")
			packet_id = data[0].decode('utf-8')
			print(packet_id)
			if int(data[1].decode('utf-8')) != -1:
					self.sender_size = int(data[1].decode('utf-8'))
			if packet_id != 'END':
				packet_id = int(packet_id)
				sender_multiple = packet_id // self.window_size #for timeout stuff
				
				#if sender_multiple <= self.multiple: #if a com was dropped if something else dropped just redo the stuff

				if packet_id not in self.collected_packets:
					self.window[packet_id%self.window_size-self.cum_index] = data[2]
					
					#print(self.tot_packets)
					self.tracking_pack[packet_id%self.window_size-self.cum_index] = packet_id 
					self.curr_packet = packet_id
					self.collected_packets.append(packet_id)
				if self.curr_packet > self.max_packet:
					self.max_packet = self.curr_packet
				#print(packet_id)
				#print(f"nack timer living {self.nack_timer_live}")
				nack_timer_live = self.nack_timer_live
				self.nack_timer_live = nack_timer_live
				if self.nack_timer_live == 0:
					self.timer_init()
					self.nack_timer_live = 1
				# else: #this is if a com is dropped
				# 	print("send com from timeout")
				# 	msg = (f"COM|{self.multiple}")
				# 	msg = msg.encode('utf-8')
				# 	for i in range(2):
				# 		recv_monitor.send(sender_id, msg)
					

if __name__ == '__main__':
	#print("Receivier starting up!")
	config_path = sys.argv[1]

	# Initialize sender monitor
	recv_monitor = Monitor(config_path, 'receiver')
	
	# Parse config file
	cfg = configparser.RawConfigParser(allow_no_value=True)

	cfg.read(config_path)
	sender_id = int(cfg.get('sender', 'id'))
	file_to_send = cfg.get('nodes', 'file_to_send')
	write_location = cfg.get('receiver','write_location')
	max_packet_size = int(cfg.get('network', 'MAX_PACKET_SIZE'))
	window_size = int(cfg.get('sender', 'window_size'))

	packet_list, tot_packets = divide_packets(file_to_send, max_packet_size-12) #removed -8
	recieve_packets = packet_reciever(sender_id,tot_packets,write_location,max_packet_size, recv_monitor, window_size)
	recieve_packets.start_up()
	#print(tot_packets)
	''' I could add something that checks that the packets are the same'''

	# 	#print(type(packet))
		
	# with open(write_location, 'w') as f:
	# 	f.write(to_write.decode('utf-8'))
	# recv_monitor.recv_end(write_location, sender_id)
	# #print(f"Ending: {time.time()}")
	# msg = 'END'
	# msg = msg.encode('utf-8')
	# recv_monitor.send(sender_id, msg)
	

