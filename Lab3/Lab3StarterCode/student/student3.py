from typing import List

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

################ MY CODE

import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

THROUGHPUT_IN_SIZE = 5
THROUGHPUT_OUT_SIZE = 5
TRAIN_PER_STEP = 3
PLOT = False
LOAD_MODEL = True
SAVE_MODEL = True
MODEL_PTH = os.path.join('models', 'mithru.pth')

class Prediction_Buffer():
	def __init__(self):
		self.throughputs = []
		self.buffer_capacities = []

	def add_data(self, client_message, throughput_prediction, first=False):
		if not first:
			self.throughputs.append(client_message.previous_throughput)
		self.buffer_capacities.append(client_message.buffer_seconds_until_empty)

	def sample_random(self):
		rand_idx = int((len(self.throughputs) - (THROUGHPUT_IN_SIZE + THROUGHPUT_OUT_SIZE)) * random.random())

		prior_throughputs = self.throughputs[rand_idx:rand_idx+THROUGHPUT_IN_SIZE]
		next_throughputs = self.throughputs[rand_idx+THROUGHPUT_IN_SIZE:rand_idx+THROUGHPUT_IN_SIZE+THROUGHPUT_OUT_SIZE]

		return torch.tensor(prior_throughputs), torch.tensor(next_throughputs)
	
	def get_most_recent_throughputs(self):
		return torch.tensor(self.throughputs[-THROUGHPUT_IN_SIZE:])
	
	def __len__(self):
		return max(0, len(self.throughputs) - (THROUGHPUT_IN_SIZE + THROUGHPUT_OUT_SIZE))
	

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(output_size, hidden_size)
        self.output_size = output_size
        
    def forward(self, input_seq, target_seq_len):
        input_seq = input_seq.unsqueeze(0).unsqueeze(-1)
        hidden, cell = self.encoder(input_seq)
        # Start with an initial input (usually zero) to the decoder
        decoder_input = torch.zeros((input_seq.size(0), 1, self.output_size), device=input_seq.device)
        outputs = []
        
        for _ in range(target_seq_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(output)
            decoder_input = output.unsqueeze(1)
        
        return torch.cat(outputs, dim=1)

# Parameters
input_size = 1  # Size of each input item
output_size = 1  # Size of each output item
hidden_size = 128  # Number of features in the hidden state

# Model initialization
model = Seq2Seq(input_size, output_size, hidden_size)
if LOAD_MODEL:
	if not os.path.exists(MODEL_PTH):
		torch.save(model.state_dict(), MODEL_PTH)
	model.load_state_dict(torch.load(MODEL_PTH))
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
	
prediction_buffer = Prediction_Buffer()

# Your helper functions, variables, classes here. You may also write initialization routines to be called
# when this script is first imported and anything else you wish.

def map_to_nearest_option(value, options):
	min_dist = float('inf')
	closest_option = None
	for options in options:
		dist = abs(options - value)
		if dist < min_dist:
			min_dist = dist
			closest_option = options
	return closest_option

def extract_bitrate(tensor, bitrate_options, V):
	tensorlist = tensor.detach().tolist()[0]
	return V * map_to_nearest_option(tensorlist[0], bitrate_options), tensorlist[0]

def calculate_QoE(selected_Qs, buffer_capacities, throughputs, client_message):
	avg_Q = sum(selected_Qs) / len(selected_Qs)
	var_Q = 0
	for i in range(len(selected_Qs)-1):
		var_Q += abs(selected_Qs[i] - selected_Qs[i+1])
	buffering = 0 
	for i in range(len(selected_Qs)):
		buffering += max(selected_Qs[i] / throughputs[i] - buffer_capacities[i], 0)
	
	return client_message.quality_coefficient * avg_Q + client_message.variation_coefficient * var_Q + client_message.rebuffering_coefficient

start = [True]

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

	if len(prediction_buffer) > 1:
		for i in range(TRAIN_PER_STEP):
			optimizer.zero_grad()
			inputThroughputs, observedThroughputs = prediction_buffer.sample_random()
			predictedThroughputs = model(inputThroughputs, THROUGHPUT_OUT_SIZE)
			loss = criterion(predictedThroughputs, observedThroughputs)
			loss.backward()
			optimizer.step()

		currentThroughputs = prediction_buffer.get_most_recent_throughputs()
		upcomingThroughputPredictions = model(currentThroughputs, THROUGHPUT_OUT_SIZE)
		selected_bitrate, throughput_prediction = extract_bitrate(upcomingThroughputPredictions, client_message.quality_bitrates, client_message.buffer_seconds_per_chunk)
	else:
		selected_bitrate = client_message.quality_bitrates[1]
		throughput_prediction = selected_bitrate / client_message.buffer_seconds_per_chunk

	if start[0]:
		start[0] = False
	else:
		prediction_buffer.add_data(client_message, throughput_prediction)

	if len(client_message.upcoming_quality_bitrates) == 0:
		if SAVE_MODEL:
			torch.save(model.state_dict(), MODEL_PTH)

	return client_message.quality_bitrates.index(selected_bitrate)