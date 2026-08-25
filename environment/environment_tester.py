from .environment import RoutingEnvironment
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def getPathLength(path):
    dist = 0
    for nodeIdx in range(len(path)-1):
        #print(f'Edge dist between {path[nodeIdx]} and {path[nodeIdx+1]}: {G.get_edge_data(path[nodeIdx], path[nodeIdx+1])}')
        dist = dist + G.get_edge_data(path[nodeIdx], path[nodeIdx+1])['distance']
    return(dist)

env = RoutingEnvironment()

T = range(100)
shortest_path_dist = []
loss = []
delay_tot = []
delay_path = []
delay_queue = []

sat0_0_queue_delay = []
for t in T:
    G = env.snapshot()
    sat0_0_queue_delay.append(G.nodes(data=True)['SAT0-0']['queue_delay'])

    shortest_path = tuple(nx.shortest_path(G, source='AC-0', target='TGT-0', weight='distance'))
    #print(f'Shortest Path {shortest_path} has distance {getPathLength(shortest_path)}')
    shortest_path_dist.append(getPathLength(shortest_path))
    overlays = env.get_overlays()
    underlays = []
    for overlay in overlays:
        underlays.append(env.overlay_to_underlay(overlay))

        underlay_path_len = getPathLength(underlays[-1])
        path_metrics = env.path_metrics(underlays[-1])

        #print(round(100*path_metrics[2]))
        #print(round(underlay_path_len))
        assert round(100*path_metrics[2]) == round(underlay_path_len) #propagation speed is constant (100)
        #print(f'Path {underlays[-1]} has distance {getPathLength(underlays[-1])}')


    assert(shortest_path in underlays)
    path_metrics = env.path_metrics(shortest_path) #tot_delay=delay_queue+delay_path, 1.0 - keep, delay_path, delay_queue
    assert(path_metrics[0] == path_metrics[2] + path_metrics[3])
    assert(0 <= path_metrics[1] and path_metrics[1] <=1)
    loss = path_metrics[1]
    delay_path.append(path_metrics[2])
    delay_queue.append(path_metrics[3])
    delay_tot.append(path_metrics[0])

    env.step()

fig, ax1 = plt.subplots()

line1 = ax1.plot(T, delay_path, label="Path delay")
ax1.set_xlabel("T")

line2 = ax1.plot(T, delay_queue, linestyle=':', color='red', label="Queue delay")
ax1.set_ylabel("Delay")

lines = line1 + line2
labels = [line.get_label() for line in lines]

plt.title("Delay path from aircraft to target")
plt.show()


fig, ax1 = plt.subplots()

#line1 = ax1.plot(T, sat0_0_queue_delay, label="Queue delay - SAT0-0")
#ax1.set_xlabel("T")
#ax1.set_ylabel("SAT0-0 Queue Delay")
#
#plt.title("Queue delay - SAT0-0")
#plt.show()