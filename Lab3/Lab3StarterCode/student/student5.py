from typing import List
from matplotlib import pyplot as plt

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
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random

THROUGHPUT_IN_SIZE = 7
THROUGHPUT_OUT_SIZE = 5
TRAIN_PER_STEP = 3
PLOT = False
LOAD_MODEL = True
SAVE_MODEL = True
MODEL_PTH = os.path.join('models', 'BB.pth')


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

		prior_throughputs = self.throughputs[rand_idx:rand_idx+THROUGHPUT_IN_SIZE]
		next_throughputs = self.throughputs[rand_idx+THROUGHPUT_IN_SIZE:rand_idx+THROUGHPUT_IN_SIZE+THROUGHPUT_OUT_SIZE]
		buffers = self.buffer_capacities[rand_idx:rand_idx+THROUGHPUT_IN_SIZE]

		return prior_throughputs, next_throughputs, buffers
	
	def get_most_recent_throughputs(self):
		return self.throughputs[-THROUGHPUT_IN_SIZE:], self.buffer_capacities[-THROUGHPUT_IN_SIZE:]
	
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
output_size = 1 # Size of each output item
hidden_size = 128  # Number of features in the hidden state

# Model initialization
model = Seq2Seq(input_size, output_size, hidden_size)
if LOAD_MODEL:
	if not os.path.exists(MODEL_PTH):
		torch.save(model.state_dict(), MODEL_PTH)
	model.load_state_dict(torch.load(MODEL_PTH))

optimizer = optim.Adam(model.parameters(), lr=0.01)
	
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

def map_output_tensor_to_nearest_option(tensor, options_series):
	diff = []
	bitrate_list = tensor.clone().detach().tolist()[0]
	for bitrate in bitrate_list:
		mapped_bitrate = map_to_nearest_option(bitrate, options_series)
		# map = bitrate + diff
		# map - bitrate = diff
		diff.append(mapped_bitrate - bitrate)
	return torch.tensor(diff, requires_grad=True).view(tensor.size())


def extract_bitrate(tensor, bitrate_options):
	tensorlist = tensor.detach().tolist()
	return map_to_nearest_option(tensorlist[0][0], bitrate_options)

def tensor_to_float(tensor):
	return float(tensor.detach())

def tensor_QoE(selected_bitrates, throughputs_list, buffer, V, client_message):
	avg_Q = torch.sum(selected_bitrates)
	var_Q = torch.sum(torch.abs(selected_bitrates[1:] - selected_bitrates[:-1]))
	
	bitrate_list = selected_bitrates.clone().detach().tolist()[0]
	download_times_list = [bitrate/throughput for bitrate, throughput in zip(bitrate_list, throughputs_list)]
	buffers_list = [buffer]
	for download_time in download_times_list:
		new_buffer = max(buffers_list[-1] - download_time + V, 0)
		buffers_list.append(new_buffer)
	buffers = torch.tensor(buffers_list[:-1]).view(selected_bitrates.size())

	throughputs = torch.tensor(throughputs_list).view(selected_bitrates.size())

	# download time - buffer
	#If this value is greater than 0, rebuffering is occuring
	#if it is negative, the buffer is refilling
	download_times = selected_bitrates / throughputs
	buffering = download_times - buffers
	# if (buffering > 0).any():
	# 	print(f"DL {download_times}\n - BUFF {buffers}\n = BUFFERING {buffering}")
	buffering = torch.where(buffering >= 0, buffering, torch.tensor(0.0, requires_grad=True))
	buffering = torch.sum(buffering)

	QoE = client_message.quality_coefficient * avg_Q - 0*client_message.variation_coefficient * var_Q - client_message.rebuffering_coefficient * buffering
	return QoE, avg_Q, var_Q, buffering

if PLOT:
	# Enable interactive mode
	os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
	plt.ioff()
	plt.ion()

	# Create a figure and an axes.
	fig, ax = plt.subplots()

	# Initialize three line objects (one in each plot)
	line1, = ax.plot([], [], label='Avg_Q')  # Line for the first dataset
	line2, = ax.plot([], [], label='Var_Q')  # Line for the second dataset
	line3, = ax.plot([], [], label='Buffer')  # Line for the third dataset
	ax.legend()

	# Setting the axes limits
	# ax.set_xlim(0, 100)
	# ax.set_ylim(0, 15)

	# Data lists for each line
	x = [0]
	x_data = []
	y_data1 = []
	y_data2 = []
	y_data3 = []

# Update the plot with new data
def update_plot(y1, y2, y3):
    y1 = tensor_to_float(y1)
    y2 = tensor_to_float(y2)
    y3 = tensor_to_float(y3)
	
    x_data.append(x[0])
    x[0] += 1
    y_data1.append(y1)
    y_data2.append(y2)
    y_data3.append(y3)
    line1.set_xdata(x_data)
    line1.set_ydata(y_data1)
    line2.set_xdata(x_data)
    line2.set_ydata(y_data2)
    line3.set_xdata(x_data)
    line3.set_ydata(y_data3)


def draw_plot():
    ax.relim()  # Recompute the ax.dataLim
    ax.autoscale_view()  # Update ax.viewLim using the new dataLim
    
    fig.canvas.draw()
    fig.canvas.flush_events()

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
			inputThroughputs, observedThroughputs, buffers = prediction_buffer.sample_random()
		
			input = torch.tensor(buffers).unsqueeze(0).unsqueeze(-1)
			selected_bitrates = model(input, THROUGHPUT_OUT_SIZE)
			#print(selected_bitrates, "\n", client_message.quality_bitrates)
			diff = map_output_tensor_to_nearest_option(selected_bitrates, client_message.quality_bitrates)
			#print(diff)
			#selected_bitrates += diff
			#print(selected_bitrates)
			QoE, avg_q, var_q, buffering = tensor_QoE(selected_bitrates, observedThroughputs, buffers[-1], client_message.buffer_seconds_per_chunk, client_message)
			if PLOT:
				update_plot(avg_q, var_q, buffering)
			(-QoE).backward()
			optimizer.step()
		if PLOT:
			draw_plot()

		currentThroughputs, currentBuffers = prediction_buffer.get_most_recent_throughputs()
		input = torch.stack((torch.tensor(currentThroughputs), torch.tensor(currentBuffers)), dim=1)
		input = input.unsqueeze(0)
		upcomingThroughputPredictions = model(input, THROUGHPUT_OUT_SIZE)
		selected_bitrate = extract_bitrate(upcomingThroughputPredictions, client_message.quality_bitrates)
	else:
		selected_bitrate = client_message.quality_bitrates[0]

	if start[0]:
		start[0] = False
	else:
		prediction_buffer.add_data(client_message) 
	
	#IF this is the last endpoint
	if len(client_message.upcoming_quality_bitrates) == 0:
		if SAVE_MODEL:
			torch.save(model.state_dict(), MODEL_PTH)
	

	return client_message.quality_bitrates.index(selected_bitrate)