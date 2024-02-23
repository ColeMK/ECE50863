#!/usr/bin/env python3
from monitor import Monitor
import sys
import time
# Config File
import zlib
import configparser
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

	packet_list, tot_packets = divide_packets(file_to_send, max_packet_size-8) #removed -8
	#print(tot_packets)
	''' I could add something that checks that the packets are the same'''
	curr_packet = 0
	to_write = b''
	#to_write = to_write.encode('utf-8')
	# Exchange messages!

	#print(curr_packet)
	ids_recieved = []
	collected_packets = []
	while curr_packet != (tot_packets): # may be -1
		addr, data = recv_monitor.recv(max_packet_size)
		# try:
		# 	data = data.decode('utf-8', errors="ignore")
		# except UnicodeDecodeError as e:
		# 	raise Exception("Error decoding byte string:", e)
		#print(f"Data: {len(data)}")
		data = data.split(b'|',1) #would need to use different split if it was in one of those files
		#print(f"dataa length: {len(data)}")
		data[0] = int(data[0].decode('utf-8'))
		#print(data[0])
		if int(data[0]) == curr_packet:
			ids_recieved.append(data[0])
			msg = f"ACK|{data[0]}"
			#print(f"Sendign ACK: {msg}")
			msg = msg.encode('utf-8')
			
			recv_monitor.send(sender_id, msg) #I dont think I need the b they had in their code
			#to_write = to_write + data[1]
			collected_packets.append(data[1])
			#f.write(data[1])
			curr_packet += 1
		elif (int(data[0])<curr_packet):
			curr_packet = int(data[0])
			msg = f"ACK|{curr_packet}" #ACk lower packets
			msg = msg.encode('utf-8')
			recv_monitor.send(sender_id, msg)
			collected_packets = collected_packets[:data[0]]
			collected_packets.append(data[1])
			curr_packet+=1
		else: #data[0] in ids_recieved:#already recieved but out of order or ack drop
			msg = f"ACK|{curr_packet}" #Ack the same packet again
			#print(f"already recieved,out order, or mustve dropped: {msg}")
			msg = msg.encode('utf-8')
			recv_monitor.send(sender_id, msg) #I dont think I need the b they had in their code
	# Exit! Make sure the receiver ends before the sender. send_end will stop the emulator.
	for packet in collected_packets:
		to_write = to_write + packet
		#print(type(packet))
		
	with open(write_location, 'w') as f:
		f.write(to_write.decode('utf-8'))
	recv_monitor.recv_end(write_location, sender_id)
	#print(f"Ending: {time.time()}")
	msg = 'END'
	msg = msg.encode('utf-8')
	recv_monitor.send(sender_id, msg)
	

