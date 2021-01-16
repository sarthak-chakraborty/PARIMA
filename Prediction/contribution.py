# Code to pick up framewise object and viewport contributions from the file and find average contributions
import sys

topic = sys.argv[1]

f = open(topic + '_contribution.txt', 'r')
contri_list = f.readlines()
contri_list = [x.split(' ') for x in contri_list]
contri_list = [[float(y) for y in x] for x in contri_list]

contri_dict = dict()
count_dict = dict()

for x in contri_list:
	frame = int(x[0])
	if count_dict.get(frame, None) is None:
		count_dict[frame] = 1
		contri_dict[frame] = [x[1], x[2]]
	else:
		# Find average contributions
		count_dict[frame] += 1
		contri_dict[frame][0] = (contri_dict[frame][0] * (count_dict[frame] - 1) + x[1])/count_dict[frame]
		contri_dict[frame][1] = (contri_dict[frame][1] * (count_dict[frame] - 1) + x[2])/count_dict[frame]  

avg_contri_list = []
for x in contri_dict:
	avg_contri_list.append([x, contri_dict[x][0], contri_dict[x][1]])    # Append to file

f.close()
f = open(topic + '_avg_contribution','w')

for x in avg_contri_list:
	vp = x[1]/(x[1]+x[2])                                         # Normalise contributions to sum 1
	obj = 1-vp
	f.write(str(x[0]) + ' ' + str(vp) + ' ' + str(obj) + '\n')

f.close()