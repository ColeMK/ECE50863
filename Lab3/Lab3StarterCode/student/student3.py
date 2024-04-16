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

THROUGHPUT_IN_SIZE = 10
THROUGHPUT_OUT_SIZE = 5
TRAIN_PER_STEP = 3

class Prediction_Buffer():
	def __init__(self):
		self.throughputs = []
		self.buffer_capacities = []

	def add_data(self, client_message, first=False):
		if not first:
			self.throughputs.append(client_message.previous_throughput)
		self.buffer_capacities.append(client_message.buffer_seconds_until_empty)

	def sample_random(self):
		rand_idx = int((len(self.throughputs) - (THROUGHPUT_IN_SIZE + THROUGHPUT_OUT_SIZE)) * random.random())
		print(rand_idx, len(self.throughputs), len(self))

		prior_throughputs = self.throughputs[rand_idx:rand_idx+THROUGHPUT_IN_SIZE]
		next_throughputs = self.throughputs[rand_idx+THROUGHPUT_IN_SIZE:rand_idx+THROUGHPUT_IN_SIZE+THROUGHPUT_OUT_SIZE]

		return torch.tensor(prior_throughputs).unsqueeze(0), torch.tensor(next_throughputs).unsqueeze(0)
	
	def get_most_recent_throughputs(self):
		return torch.tensor(self.throughputs[-THROUGHPUT_IN_SIZE:])
	
	def __len__(self):
		return max(0, len(self.throughputs) - (THROUGHPUT_IN_SIZE + THROUGHPUT_OUT_SIZE))
	

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(2)
		#(1,10,1)

		# Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).to(x.device)
        x = x.permute(1,0,2)
        print(x.size(), h0.size(), c0.size())

        # Forward pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Passing the output of the last time step to the fully connected layer
        out = self.fc(out[:, -1, :])
        return out
	

class GRUnet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, num_layers, drop_prob=0.2):
                super(GRUnet, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.gru = nn.GRU(input_size, hidden_size, num_layers)
                self.fc = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()
                self.softmax = nn.Softmax(dim=1)
                
            def forward(self, x, h):
                x = x.unsqueeze(0)
                print(x.size())
                out, h = self.gru(x, h)
                out = self.fc(self.relu(out[:,-1]))
                out = self.softmax(out)
                return out, h

            def init_hidden(self):
                weight = next(self.parameters()).data
                #                                     batch_size   
                hidden = weight.new(  self.num_layers,     1,         self.hidden_size   ).zero_()
                return hidden
	
prediction_buffer = Prediction_Buffer()

model = GRUnet(input_size=THROUGHPUT_IN_SIZE, hidden_size=50, output_size=THROUGHPUT_OUT_SIZE, num_layers=1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
hidden = [model.init_hidden()]

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
			predictedThroughputs, hidden[0] = model(inputThroughputs, hidden[0])
			loss = criterion(predictedThroughputs, observedThroughputs)
			loss.backward()
			optimizer.step()

		currentThroughputs = prediction_buffer.get_most_recent_throughputs()
		upcomingThroughputPredictions = model(currentThroughputs)
		predictionBitrate = upcomingThroughputPredictions[0] * client_message.buffer_seconds_per_chunk
		selected_bitrate = map_to_nearest_option(predictionBitrate, [bitrate for bitrate in client_message.quality_bitrates if bitrate < predictionBitrate])
	else:
		selected_bitrate = client_message.quality_bitrates[0]

	if start[0]:
		start[0] = False
	else:
		prediction_buffer.add_data(client_message)

	

	return client_message.quality_bitrates.index(selected_bitrate)