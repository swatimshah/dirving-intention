import mne
import numpy

file = mne.io.read_raw_eeglab("EventsComplete1.set")
complete_data = file.to_data_frame()
complete_data.to_csv('complete_data.csv', index=False)
loaded_complete_data = numpy.loadtxt('complete_data.csv', delimiter=',', skiprows=1)

events = mne.events_from_annotations(file)
print(events)
print(events[0].shape)
print(len(events[0]))
lengthOfEvents = len(events[0])
events_filtered = numpy.empty((0, 3))

for i in range (0, lengthOfEvents, 1):
	#print(events[0][i])
	if (events[0][i][2] == 4):
		print(events[0][i])
		events_filtered = numpy.append(events_filtered, events[0][i].reshape(1, 3), axis=0)
	
print(events_filtered.shape)

labels_file = 'Labels.csv'
my_labels = numpy.loadtxt(labels_file, delimiter=',')
print(my_labels)
print(my_labels.shape)

combined_events_and_labels = numpy.append(events_filtered, my_labels.reshape(3, 92).transpose(), axis=1)
print(combined_events_and_labels)

combined_events_and_labels = numpy.delete(combined_events_and_labels, numpy.s_[1:3], axis=1)    

combined_events_and_labels[:, 0] = combined_events_and_labels[:, 0]/500
print(combined_events_and_labels[:, 0])

numpy.savetxt('combined_events_and_labels.csv', combined_events_and_labels, delimiter=',')

# Extract the complete data from EventsComplete1.set. 
# Pick an event from the 'combined' file. Check that event in the complete data. Extract 4 sec epoch from combined file for each event.
# Combine it with labels 

epochs = numpy.empty((0, 63))
input_data = numpy.empty((0, 63))
final_input_data = numpy.empty((0, 126000))
complete_timestamps = loaded_complete_data[:, 0]
print(len(complete_timestamps))

for i in range (92):
	input_data_gathered = 0		
	my_event_time = combined_events_and_labels[i, 0]
	for j in range (len(complete_timestamps)):	
		if (my_event_time < complete_timestamps[j]):
			for k in range (2000):
				epochs = loaded_complete_data[j + k, 1:64]
				input_data = numpy.append(input_data, epochs.reshape(1, 63), axis=0)
				input_data_gathered = 1	
			print(input_data.shape)						
			flattened_epochs = input_data.flatten().reshape(1, 126000);
			final_input_data = numpy.append(final_input_data, flattened_epochs.reshape(1, 126000), axis=0)
			if (input_data_gathered == 1):
				input_data = numpy.empty((0, 63))
				break


numpy.savetxt('final_input_data.csv', final_input_data, delimiter=',')	
