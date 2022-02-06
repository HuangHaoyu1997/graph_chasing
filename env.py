import networkx as nx
import numpy as np
import random			  
import matplotlib.pyplot as plt



class graph_chase:
    def __init__(self,num_node,edge_prob):
        self.num_node = num_node
        self.edge_prob = edge_prob
        
        self.G = None
        self.current_location = None
        self.previous_location = None
        self.target_location = None
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
    
    def reset(self,):
        # build graph
        while True:
            self.G = build_graph(self.num_node, self.edge_prob)
            number_components = nx.number_connected_components(self.G)
            largest_components = max(nx.connected_components(self.G), key=len)
            # 保证整个图是连通的
            if len(largest_components) == self.num_node:
                break
            
        # 设置node和edge的数值
        self.set_values()
        self.set_edges()
        
        # 设置智能体和目标的位置
        self.set_location()
        self.previous_location = self.current_location
        
        
        
    def update_value(self,):
        '''
        更新节点数值
        '''
        
    def step(self, action):
        pass
    
    def set_location(self,):
        node_list = list(self.G.nodes)
        self.current_location = random.choice(node_list)
        node_list.pop(self.current_location)
        self.target_location = random.choice(node_list)
        
        
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
        # print([self.G.nodes[n]['value'] for n in self.G.nodes])
    
    def set_edge(self, e, v):
        self.G.edges[e]['weight'] = v
        
    def set_edges(self,):
        for e in list(self.G.edges):
            value = np.random.rand()*2-1
            self.set_edge(e, value)
    
    def get_neighbor(self, node):
        nei = nx.neighbors(self.G, node)
        # print(self.G[node])
        return list(nei)

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
# env.set_edges()
# print(list(env.G.edges.data())[0])
# env.get_neighbor(15)
env.reset()
# env.set_location()
