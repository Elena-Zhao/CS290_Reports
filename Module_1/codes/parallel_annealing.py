import findspark
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
from networkx.readwrite import json_graph
findspark.init('/Users/zelongqiu/spark')
import pyspark
import influence_function
sc = pyspark.SparkContext()

# read graph
with open("graph/US.json", "r") as graph_data:
    graph_data = json.load(graph_data)
    NC_digraph = json_graph.node_link_graph(graph_data)

nc_N_width_nodes = [i for i in NC_digraph.nodes() if len(NC_digraph.succ[i]) >= 300]
nodes_set_broadcast = sc.broadcast(NC_digraph)
######################################################################################
#
#Influence Function Implementation
#
#######################################################################################


def cascade(init_nodes, nodes_set):#, dist_d):
    # nodes_set = nodes_set_broadcast
    action = {}
    n = len(init_nodes)
    #np.random.seed(random_d)
    #init_nodes = np.random.choice(NC_digraph.nodes(), 1)[0]
    for i in init_nodes:
        action[i] = 1
    #st = set()
    #st.add(init_nodes)
    init_list = zip([0]*len(init_nodes),init_nodes[::])
    inter = 0
    while len(init_list) != 0 and inter <10 :
        curr_node = init_list.pop(0)
        #print curr_node
        for i in nodes_set[curr_node[1]]:
            if i not in action:
                b = nodes_set.node[i]['review_count']
                a =  nodes_set[curr_node[1]][i]['weight']
                #np.random.seed(12)
                b_dist = np.sqrt(np.random.beta(a = a, b = b))
                #np.random.seed(12)
                u_dist = np.random.uniform(0,1)
                if b_dist > u_dist:
                    action[i] = 1
                    #st.add(i)
                    inter = curr_node[0] + 1
                    init_list.append((inter, i))
                    n = n + 1
    return n

def influence_function(N, init_nodes, partition_num, nodes_set_broadcast):
    nodes_set = nodes_set_broadcast.value
    activated_num_rdd = sc.parallelize([init_nodes]*N, partition_num)
    activated_num = activated_num_rdd.map(lambda x: cascade(x, nodes_set)).reduce(lambda x, y: x+y)
    return activated_num/N
####################################################################################


######################################################################################
#
#Simulated Annealing Algorithm Implementation
#
######################################################################################

def next_step(X, nodes_set):
    tmp = X[0]
    while tmp in X:
        tmp = np.random.choice(nodes_set)
    X[np.random.choice(range(len(X)))] = tmp
    return X


def simulated_annealing_rst(function, NC_digraph, initial_X, initial_temp, beta, iterr, nc_N_width_nodes):
    
    accepted = 0
    unaccept = 0
    #X = initial_X.copy()
    X = initial_X[::]
    T = initial_temp
    
    history = list()
    # Evaluate E
    prev_E = function(X, NC_digraph)
    history.append(prev_E)
    
    for i in xrange(iterr):
        # Propose new path.
        X_star = next_step(X, nc_N_width_nodes)
        # Evaluate E
        new_E = function(X_star, NC_digraph)
        delta_E = new_E - prev_E
        
        # Flip a coin
        U = np.random.uniform()
        unaccept = unaccept + 1
        if U < np.exp(delta_E / T):
            print "iteration", i
            unaccept = 0
            accepted += 1
            history.append(new_E)
            print 'value:', new_E
            # Copy X_star to X
            #X = X_star.copy()
            X = X_star[::]
            print 'set:', X
            prev_E = new_E
            # cool down the temperature very slowly
            T = T/(1+(beta*T))
        if unaccept == 300:
            T = T * 1.1

    return X, history


# Simulated Annealing Parameters
#initial_X = np.random.choice(nc_N_width_nodes, 3) # Start random
initial_X = [u'Yw-Q_4QrwWffjnHWLvo4kw', u'DcgO7qiYKS2VuAJj2dQpcg', u'EGiAtB4sgZhDdYpRkDneig']
initial_temp = 3000.
beta = 0.0005
iterr = 1500
N = 100
partition_num = 4

start_time = time.time()
solution, history = simulated_annealing_rst(cascade, NC_digraph, initial_X,
                                            initial_temp, beta, iterr,nc_N_width_nodes)

print "path:", solution
print "length:", len(history)
print "Time used:", time.time() - start_time
plt.figure(figsize=(12, 8))
plt.plot(history)
plt.title("History")
plt.ylabel("$f(x_1,x_2)$",fontsize=12)
plt.xlabel("Accepted", fontsize=12)
plt.show()
