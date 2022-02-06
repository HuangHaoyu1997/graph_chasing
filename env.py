import networkx as nx
import numpy as np
import random			  
import matplotlib.pyplot as plt



class graph_chase:
    def __init__(self,num_node,edge_prob):
        self.num_node = num_node
        self.edge_prob = edge_prob
        
        # build graph
        while True:
            self.G = build_graph(num_node, edge_prob)
            number_components = nx.number_connected_components(self.G)
            largest_components = max(nx.connected_components(self.G), key=len)
            # 保证整个图是连通的
            if len(largest_components) == self.num_node:
                break
        self.set_values()
        
        
        # nx.degree(self.G)
        # DVweight = self.G.degree()
        # degree_sum = sum(span for n, span in DVweight) 		#各节点度数之和
        # degree_max = max(span for n, span in DVweight)		#节点最大度数

        # print("度数之和: " + str(degree_sum))
        # print("节点最大度数：" + str(degree_max))
        # print("最大连通子图:" + str(largest_components))
        # print("最大连通子图长度："+ str(len(largest_components)))
        # print("连通子图个数: "+str(nx.number_connected_components(self.G)))
        # nx.draw_networkx(G, with_labels=True)
        # plt.show(
    def step(self,action):
        pass
    
    def set_value(self, n, v):
        '''
        set the value for node n
        '''
        self.G.nodes[n]['value'] = v
        # print(self.G.nodes[node])
    
    def set_values(self,):
        '''
        set values for all nodes
        '''
        for node in self.G.nodes:
            value = np.random.rand()
            self.set_value(node, value)
        print([self.G.nodes[n]['value'] for n in self.G.nodes])
    
    def set_edge(self, ):
        print(self.G.edges[0][1])

def build_graph(num_node, prob):
    '''
    num_node
    prob: edge概率
    '''
    G = nx.Graph()
    H = nx.path_graph(num_node) # 利用path_graph添加节点，不用path_graph中的edge，path_graph即链图
    G.add_nodes_from(H)

    def rand_edge(vi,vj,p):
        probability =random.random()
        if(probability<p):
            G.add_edge(vi,vj)   
    i=0
    while (i<num_node):
        j=0
        while(j<i):
                rand_edge(i,j,p=prob)
                j +=1
        i +=1
    return G



env = graph_chase(num_node=20, edge_prob=0.4)
env.set_edge()
