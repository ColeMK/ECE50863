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
		self.nack_timer = 0.2
		self.collected_packets = []
		self.end = 0
		self.multiple = 0
		self.final_packet = 0
		self.final = 0
		self.sender_id = sender_id
		self.tot_packets = tot_packets
		self.write_location = write_location
		self.max_packet_size = max_packet_size
		self.recv_monitor = recv_monitor
		self.sender_size = window_size
		self.max_packet = 0
		self.cum_index = 0
		self.nack_index = 0 
		self.written_packets = []
		#self.last_tracking = -1
		self.file = open(self.write_location, 'wb')
		self.write_thread = threading.Thread(target=self.write_to_file, args=(self.window, self.write_location))
		self.timer_thread = threading.Thread(target=self.send_nack)
		self.nack_timer_live = 0

	def write_to_file(self, buff_window, write_location):
		#with open(write_location, 'ab') as f:
		for i,p in enumerate(buff_window):
			##print(i)
			
			if p!= -1 and self.tracking_pack_buff[i] not in self.written_packets:
				#print(f"wrote {self.tracking_pack_buff[i]}")
				#if self.self.tracking_pack_buff[i] != self.last_tracking + 1:
				self.written_packets.append(self.tracking_pack_buff[i])
				self.file.write(p)

	def timer_init(self):
		self.timer_thread = threading.Timer(self.nack_timer, self.send_nack)
		self.timer_thread.start()

	def send_nack(self): #might send twice for redundancy
		#self.nack_timer_live = 0
		indexes_of_minus_one = [index for index, value in enumerate(self.window[0:self.sender_size]) if value == -1]
		if len(indexes_of_minus_one) > 0:
			self.nack_index = indexes_of_minus_one[0]
		else: 
			self.nack_index = self.window_size
		if self.nack_index != 0:
			before = self.tracking_pack[self.nack_index-1]
		else:
			before = -1

		msg = (f"N|{before}|{indexes_of_minus_one}")
		##print(len(msg),msg)
		msg = msg.encode('utf-8')
		#print(f"before is {before}")
		for i in range(2):
			#print("send_nack")
			recv_monitor.send(sender_id, msg)
		# may switch order of this
		buff_window = self.window[0:self.nack_index]
		self.tracking_pack_buff = self.tracking_pack
		if self.write_thread.is_alive():
			self.write_thread.join()
		self.write_thread = threading.Thread(target=self.write_to_file, args=(buff_window, self.write_location))
		self.write_thread.start()
		#if statement for if no nacks
		self.window = self.window[self.nack_index:] + [-1] * self.nack_index #takes it back to proper size
		self.processed = 0
		#print(f"Widnow length: {len(self.window)}")
		
		self.tracking_pack = self.tracking_pack[self.nack_index:] + [-1] * self.nack_index
		self.cum_index = (self.cum_index + self.nack_index) % self.window_size
		#print(self.final)
		#if self.final == 1:
			#print(f"tracking pack: {self.tracking_pack}")
		if self.final == 1 and self.final_packet not in self.tracking_pack:
			self.write_thread.join()
			self.file.close()
			recv_monitor.recv_end(self.write_location, self.sender_id)
			#print(f"Ending: {time.time()}")
			self.end = 1
			#print("END")
			while(True):
				msg = 'E'
				msg = msg.encode('utf-8')
				recv_monitor.send(sender_id, msg)
				time.sleep(.01)
			
		# self.timer_init()
		self.nack_timer_live = 0
	
	def start_up(self):
		#self.timer_thread = threading.Timer(self.nack_timer, self.send_nack())
		#self.send_com_thread = threading.Thread(target = self.send_com)
		#self.send_com_thread.start()
		flag = 0
		while self.end != 1: # may be -1
			
			addr, data = recv_monitor.recv(max_packet_size)
			if flag == 0:
				past_time = time.time()
				flag = 1
			# try:
			# 	data = data.decode('utf-8', errors="ignore")
			# except UnicodeDecodeError as e:
			# 	raise Exception("Error decoding byte string:", e)
			##print(f"Data: {len(data)}")

			data = data.split(b'|',2) #would need to use different split if it was in one of those files
			##print(f"dataa length: {len(data)}")
			self.packet_id = data[0].decode('utf-8')
			#print(self.packet_id)
			if self.packet_id != 'END':
				self.packet_id = int(self.packet_id)
				
				sender_multiple = self.packet_id // self.window_size #for timeout stuff
				#if sender_multiple <= self.multiple: #if a com was dropped if something else dropped just redo the stuff
				if int(data[1].decode('utf-8')) != -1:
					self.sender_size = int(data[1].decode('utf-8'))
				self.window[self.packet_id%self.window_size-self.cum_index] = data[2]
				
				##print(self.tot_packets)
				self.tracking_pack[self.packet_id%self.window_size - self.cum_index] = self.packet_id 
				self.curr_packet = self.packet_id
				if self.curr_packet > self.max_packet:
					self.max_packet = self.curr_packet
					if self.max_packet == tot_packets - 1:
						self.final_packet = self.max_packet
						self.final = 1
						
				##print(packet_id)
				##print(f"nack timer living {self.nack_timer_live}")
				nack_timer_live = self.nack_timer_live
				self.nack_timer_live = nack_timer_live
				# if self.nack_timer_live == 0:
				# 	if self.timer_thread.is_alive():
				# 		self.timer_thread.join()
				# 	self.timer_init()
				# 	self.nack_timer_live = 1
			if (time.time()> past_time + self.nack_timer):
				indexes_of_minus_one = [index for index, value in enumerate(self.window[0:self.sender_size]) if value == -1]
				if len(indexes_of_minus_one) > 0:
					self.nack_index = indexes_of_minus_one[0]
				else: 
					self.nack_index = self.window_size
				if self.nack_index != 0:
					before = self.tracking_pack[self.nack_index-1]
				else:
					before = -1

				msg = (f"N|{before}|{indexes_of_minus_one}")
				##print(len(msg),msg)
				msg = msg.encode('utf-8')
				#print(f"before is {before}")
				for i in range(2):
					#print("send_nack")
					recv_monitor.send(sender_id, msg)
				# may switch order of this
				buff_window = self.window[0:self.nack_index]
				self.tracking_pack_buff = self.tracking_pack
				if self.write_thread.is_alive():
					self.write_thread.join()
				self.write_thread = threading.Thread(target=self.write_to_file, args=(buff_window, self.write_location))
				self.write_thread.start()
				#if statement for if no nacks
				self.window = self.window[self.nack_index:] + [-1] * self.nack_index #takes it back to proper size
				self.processed = 0
				#print(f"Widnow length: {len(self.window)}")
				
				self.tracking_pack = self.tracking_pack[self.nack_index:] + [-1] * self.nack_index
				self.cum_index = (self.cum_index + self.nack_index) % self.window_size
				#print(self.final)
				#if self.final == 1:
					#print(f"tracking pack: {self.tracking_pack}")
				if self.final == 1 and self.final_packet not in self.tracking_pack:
					self.write_thread.join()
					self.file.close()
					recv_monitor.recv_end(self.write_location, self.sender_id)
					#print(f"Ending: {time.time()}")
					self.end = 1
					#print("END")
					while(True):
						msg = 'E'
						msg = msg.encode('utf-8')
						recv_monitor.send(sender_id, msg)
						time.sleep(.01)
				past_time = time.time()
				# self.timer_init()
				self.nack_timer_live = 0
			# else: #this is if a com is dropped
			# 	#print("send com from timeout")
			# 	msg = (f"COM|{self.multiple}")
			# 	msg = msg.encode('utf-8')
			# 	for i in range(2):
			# 		recv_monitor.send(sender_id, msg)
					

if __name__ == '__main__':
	##print("Receivier starting up!")
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
	##print(tot_packets)
	''' I could add something that checks that the packets are the same'''

	# 	##print(type(packet))
		
	# with open(write_location, 'w') as f:
	# 	f.write(to_write.decode('utf-8'))
	# recv_monitor.recv_end(write_location, sender_id)
	# ##print(f"Ending: {time.time()}")
	# msg = 'END'
	# msg = msg.encode('utf-8')
	# recv_monitor.send(sender_id, msg)
	

