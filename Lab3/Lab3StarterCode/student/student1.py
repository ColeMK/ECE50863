from typing import List
import math
# Adapted from code by Zach Peats

# ======================================================================================================================
# Do not touch the client message class!
# ======================================================================================================================


class ClientMessage:
	"""
	This class will be filled out and passed to student_entrypoint for your algorithm.
	"""
	total_seconds_elapsed: float	  # The number of simulated seconds elapsed in this test
	previous_throughput: float		  # The measured throughput for the previous chunk in kB/s

	buffer_current_fill: float		    # The number of kB currently in the client buffer
	buffer_seconds_per_chunk: float     # Number of seconds that it takes the client to watch a chunk. Every
										# buffer_seconds_per_chunk, a chunk is consumed from the client buffer.
	buffer_seconds_until_empty: float   # The number of seconds of video left in the client buffer. A chunk must
										# be finished downloading before this time to avoid a rebuffer event.
	buffer_max_size: float              # The maximum size of the client buffer. If the client buffer is filled beyond
										# maximum, then download will be throttled until the buffer is no longer full

	# The quality bitrates are formatted as follows:
	#
	#   quality_levels is an integer reflecting the # of quality levels you may choose from.
	#
	#   quality_bitrates is a list of floats specifying the number of kilobytes the upcoming chunk is at each quality
	#   level. Quality level 2 always costs twice as much as quality level 1, quality level 3 is twice as big as 2, and
	#   so on.
	#       quality_bitrates[0] = kB cost for quality level 1
	#       quality_bitrates[1] = kB cost for quality level 2
	#       ...
	#
	#   upcoming_quality_bitrates is a list of quality_bitrates for future chunks. Each entry is a list of
	#   quality_bitrates that will be used for an upcoming chunk. Use this for algorithms that look forward multiple
	#   chunks in the future. Will shrink and eventually become empty as streaming approaches the end of the video.
	#       upcoming_quality_bitrates[0]: Will be used for quality_bitrates in the next student_entrypoint call
	#       upcoming_quality_bitrates[1]: Will be used for quality_bitrates in the student_entrypoint call after that
	#       ...
	#
	quality_levels: int
	quality_bitrates: List[float]
	upcoming_quality_bitrates: List[List[float]]

	# You may use these to tune your algorithm to each user case! Remember, you can and should change these in the
	# config files to simulate different clients!
	#
	#   User Quality of Experience =    (Average chunk quality) * (Quality Coefficient) +
	#                                   -(Number of changes in chunk quality) * (Variation Coefficient)
	#                                   -(Amount of time spent rebuffering) * (Rebuffering Coefficient)
	#
	#   *QoE is then divided by total number of chunks
	#
	quality_coefficient: float
	variation_coefficient: float
	rebuffering_coefficient: float
# ======================================================================================================================


# Your helper functions, variables, classes here. You may also write initialization routines to be called
# when this script is first imported and anything else you wish.

prev_Rk = [0]
#Bk = 0
time_horizon = 5
window = 5
c_list = [0]*time_horizon
predicted_c_list = [0]*time_horizon
global total_items
total_items = 0
curr_max = -9999
buffer_fill = 0 # i need to fix this


############################ OLD CODE
# def calc_QOE(bitrate, Bk, Ct, client_message: ClientMessage, prev_rate, quality_level):
# 	chunk_qual = client_message.quality_coefficient * bitrate 
# 	variation_qual = client_message.variation_coefficient * math.fabs(bitrate - prev_rate) #maybe change to use index
# 	# clean up if my first predicted throughput is not 0
# 	if Ct != 0:
# 		rebuffer_qual = client_message.rebuffering_coefficient * max(bitrate/Ct - Bk,0) # this needs changed we want to make this 0
# 		print(rebuffer_qual)
# 	else:
# 		rebuffer_qual = 0
# 	qoe = chunk_qual - variation_qual - rebuffer_qual
# 	return qoe
# ######################################################

def calc_MAPE(predicted_c, actual_c):
	max_error = 0
	for actual, predicted in zip(actual_c, predicted_c):
		if actual != 0:
			absolute_error = abs((actual - predicted) / actual)
			max_error = max(max_error, absolute_error)
	return max_error

def robust_throughput_predictor(prev_throughput):

	if prev_throughput == 0:
		return 0
	
	c_list.append(prev_throughput)
	c_list.pop(0)
	#Calc MAPE
	max_error = calc_MAPE(predicted_c_list, c_list)
	# Calc harmonic mean divide length by reciprocals
	# Cannot handle 0s so need a special case

	# This setup current calculates the throughput from startup
	# May change to not take throughput into account at startup to build a buffer.
	count = 0
	recip_sum = 0
	for x in c_list:
		if x != 0:
			recip_sum += (1 / x)
			count += 1
	
	h_mean = count / recip_sum

	#This is the equation from the paper where we take the harmonic mean divided by MAPE
	pred_throughput = h_mean/(1+max_error)

	# append to predicted C list for MAPE
	predicted_c_list.append(pred_throughput)
	predicted_c_list.pop(0)

	return pred_throughput
#-------------------------------------------- QOE CALCS -----------------------------#
def brute_force_func(partial_dict, depth, upcoming_levels):
    global total_items
    depth += 1
    if depth == window or depth == len(upcoming_levels):
        return
    
    for k,v in partial_dict.items():
        total_items += 1
        partial_dict[k] = {value: None for value in range(len(upcoming_levels[depth]))}
        brute_force_func(partial_dict[k], depth, upcoming_levels)

def update_bk(buffer_fill, encoded_qual, Ct, seconds_per_chunk, buffer_max):
    dt = max((max(buffer_fill - ((encoded_qual)/Ct),0) + seconds_per_chunk - buffer_max),0)
    b1 = max((max(buffer_fill - ((encoded_qual)/Ct),0) + seconds_per_chunk - dt),0)
    return b1

def buffer_qual(Ct, encoded_qual, bk):
    #print(max(encoded_qual/Ct-bk,0))
    return(max(encoded_qual/Ct-bk,0))

def dict_rec(dictionary,val,prev, var, start, Ct, buffer_max, bk, buffer_comp, client_message: ClientMessage):
    for k,v in dictionary.items():
        if v != None:
            if start == 0:
                dict_rec(v, val+k, k, 0, 1, Ct, buffer_max,update_bk(buffer_fill=bk, encoded_qual=client_message.quality_bitrates[k], Ct=Ct, 
                                                    seconds_per_chunk=client_message.buffer_seconds_per_chunk,buffer_max=buffer_max), 
                                                    buffer_comp + buffer_qual(Ct,client_message.quality_bitrates[k], bk), client_message)
            else:
                #print(f"Key: {k}, var {var + abs(k-prev)}, start {start}")
                dict_rec(v, val+k, k, var + abs(k-prev), start+1, Ct, buffer_max, update_bk(buffer_fill=bk, encoded_qual=client_message.upcoming_quality_bitrates[start-1][k], Ct=Ct, 
                                                    seconds_per_chunk=client_message.buffer_seconds_per_chunk,buffer_max=buffer_max), 
                                                    buffer_comp + buffer_qual(Ct,client_message.upcoming_quality_bitrates[start-1][k], bk), client_message)
                
        
        else:
            val_score = val + k
            var_score = var + abs(k-prev)
            buffer_score = buffer_comp + buffer_qual(Ct,client_message.upcoming_quality_bitrates[start-1][k], bk)
            #print(val_score, var_score, buffer_score)
            dictionary[k] = [val_score, var_score, buffer_score, client_message.quality_coefficient*val_score-client_message.variation_coefficient*var_score-client_message.rebuffering_coefficient*buffer_score]
            
def get_max_rate(dictionary, start, final_dict, top):
    global curr_max
    for k,v in dictionary.items():
        if start == 0:
            top = k
        #print()
        if type(v) == list:
            #print(v[3])
            if v[3] > final_dict['final'][1]:
                final_dict['final'][1] = v[3]
                final_dict['final'][0] = top
                #print(final_dict)
        else:
            get_max_rate(v, start+1, final_dict, top)
#-------------------------------------------------------------------------------------------------------------#		
def student_entrypoint(client_message: ClientMessage):
	"""
	Your mission, if you choose to accept it, is to build an algorithm for chunk bitrate selection that provides
	the best possible experience for users streaming from your service.

	Construct an algorithm below that selects a quality for a new chunk given the parameters in ClientMessage. Feel
	free to create any helper function, variables, or classes as you wish.

	Simulation does ~NOT~ run in real time. The code you write can be as slow and complicated as you wish without
	penalizing your results. Focus on picking good qualities!

	Also remember the config files are built for one particular client. You can (and should!) adjust the QoE metrics to
	see how it impacts the final user score. How do algorithms work with a client that really hates rebuffering? What
	about when the client doesn't care about variation? For what QoE coefficients does your algorithm work best, and
	for what coefficients does it fail?

	Args:
		client_message : ClientMessage holding the parameters for this chunk and current client state.

	:return: float Your quality choice. Must be one in the range [0 ... quality_levels - 1] inclusive.
	"""
	# Need a basis if test horizon is not full
	# Adjustable horizon
	# List that holds previous values
	quality_levels = client_message.quality_bitrates
	upcoming_levels = client_message.upcoming_quality_bitrates
	seconds_per_chunk = client_message.buffer_seconds_per_chunk
	
	buffer_max = client_message.buffer_max_size
	buffer_size  = client_message.buffer_seconds_until_empty

	whole_dict = {}
	Ct = robust_throughput_predictor(client_message.previous_throughput)
	global buffer_fill
	#print(buffer_fill)
	global total_items
	curr_max = -9999
	#First part of initializing the dictionary for recursion
	if Ct != 0:
		if len(upcoming_levels) != 0:
			for i in range(len(quality_levels)): #because we want the index
				total_items += 1
				depth = 0
				#print(len(upcoming_levels))
				
				whole_dict[i] = {value: None for value in range(len(upcoming_levels[depth]))}
				brute_force_func(whole_dict[i], depth, upcoming_levels)
			dict_rec(dictionary=whole_dict, val=0,prev=0,var=0,start=0, Ct = Ct, buffer_max=buffer_max, bk = client_message.buffer_seconds_until_empty, buffer_comp=0, client_message=client_message)
			
			final_dict = {'final': [None, -9999]}
			top = -1
			start = 0
			get_max_rate(dictionary=whole_dict, start = start, final_dict=final_dict, top = -1)
			#print(whole_dict)
			#print(final_dict)  
		else:
			final_dict = {'final': [2,None]}



		# for i, bitrate in enumerate(client_message.quality_bitrates):
			
			

		# 	qoes[i]=calc_QOE(bitrate, buffer_size, predicted_c, client_message, prev_rate=prev_Rk[0], quality_level = i)
		
		# #print(qoes)
		
		# qual_idx = max(qoes, key=qoes.get)
		# if (client_message.buffer_seconds_until_empty - client_message.buffer_seconds_per_chunk) <= 0:
		# 	qual_idx = 0
		# #print(qoes[qual_idx])
		#prev_Rk[0] = qoes[qual_idx]
		buffer_fill = update_bk(buffer_fill, client_message.quality_bitrates[final_dict['final'][0]], Ct, seconds_per_chunk, buffer_max) #not used
		return final_dict['final'][0]  # Let's see what happens if we select the lowest bitrate every time

	else: 
            buffer_fill +=1
            return 0
