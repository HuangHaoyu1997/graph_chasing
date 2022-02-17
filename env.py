import networkx as nx
import numpy as np
import random			  
import matplotlib.pyplot as plt
from sympy import total_degree



def combinatorial(n,m):
    # 计算组合数C(n,m)=n!/((n-m)!*m!)

    def factorial(n):
    # 计算阶乘
        a = 1
        for i in range(1,n+1):
            a = a * i
        return a

    n_fac = factorial(n)
    m_fac = factorial(m)
    n_m_fac = factorial(n-m)
    return n_fac / (n_m_fac*m_fac)

class graph_chase:
    def __init__(self,num_node,edge_prob):
        self.num_node = num_node
        self.edge_prob = edge_prob
        self.step_decay = -0.01
        self.shortest_dist = 30
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
        # 建图
        while True:
            self.G = build_graph(self.num_node, self.edge_prob)
            number_components = nx.number_connected_components(self.G)
            largest_components = max(nx.connected_components(self.G), key=len)
            # 保证整个图是连通的
            if len(largest_components) >= int(0.95*self.num_node):
                break
            
        # 设置node和edge的数值
        self.set_values()
        self.set_edges()
        
        # 设置智能体和目标的位置
        self.set_location()
        self.previous_location = self.current_location
        
        # 邻接矩阵
        ad_matrix = np.array(nx.adjacency_matrix(self.G).todense())
        # 生成observation
        obs = self.get_obs()
        
        return obs
    
    def get_obs(self):
        '''
        obs是不定长矩阵
        [
            [current_loaction, value, 0]
            [target_loaction, value, 0]
            [neighbor_current, value, weight]
            ...
            [neighbor_current, value, weight]
            [neighbor_target, value, weight]
            ...
            [neighbor_target, value, weight]
        ]
        '''
        obs = [
                [self.current_location, self.G.nodes[self.current_location]['value'], 0.],
                [self.target_location, self.G.nodes[self.target_location]['value'], 0.]
                ]
        neighbor = self.get_neighbor(self.current_location)
        for i in neighbor:
            obs.append([i, self.G.nodes[i]['value'], self.G[self.current_location][i]['weight']])
        
        neighbor = self.get_neighbor(self.target_location)
        for i in neighbor:
            obs.append([i, self.G.nodes[i]['value'], self.G[self.target_location][i]['weight']])
        
        return np.array(obs)
    
    def update_value(self,):
        '''
        更新节点数值
        '''
        pass

    def is_done(self):
        if self.current_location == self.target_location:
            return True
        else:
            return False

    def step(self, action):
        neighbor = self.get_neighbor(self.current_location)
        if action in neighbor:
            self.current_location = action
            reward = self.G[self.current_location][self.previous_location]['weight'] + self.step_decay
            self.previous_location = self.current_location
        else:
            reward = self.step_decay
        
        obs = self.get_obs()
        done = self.is_done()
        info = None
        return obs, reward, done, info
    
    def set_location(self,):
        dist = 0
        node_list = list(self.G.nodes)
        self.current_location = random.choice(node_list)
        node_list.pop(self.current_location)
        while dist <= self.shortest_dist:
            
            self.target_location = random.choice(node_list)
            dist = self.get_shortest_path()
            print(dist)
        
        
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

    def get_shortest_path(self):
        '''
        计算current和target的距离
        '''
        path = nx.shortest_path(self.G, source=self.current_location, target=self.target_location)
        return len(path)


def build_graph(num_node, prob):
    '''
    num_node
    prob: edge概率
    '''
    G = nx.Graph()
    H = nx.path_graph(num_node) # 利用path_graph添加节点，不用path_graph中的edge，path_graph即链图
    G.add_nodes_from(H)
    total_edges = combinatorial(num_node,2)

    def rand_edge(vi,vj,p):
        probability =random.random()
        if(probability<p):
            G.add_edge(vi,vj)  
    count = 0
    i=0
    while (i<num_node):
        j=0
        while(j<i):
            if count <= int(0.3*total_edges):
                prob = 0.4
            elif count > int(0.3*total_edges) and count <= int(0.6*total_edges):
                prob = 0.2
            elif count > int(0.6*total_edges) and count <= total_edges:
                prob = 0.01
            rand_edge(i,j,p=prob)
            j +=1
            count += 1
        i +=1
    return G


np.random.seed(10)
random.seed(10)

env = graph_chase(num_node=200, edge_prob=0.05)
env.reset()

for _ in range(10):
    obs = env.reset()
    print(len(env.get_shortest_path()))
    done = False
    reward = 0
    step = 0
    while not done:
        action = np.random.randint(200)
        obs, r, done, info = env.step(action)
        reward += r
        step += 1
    print(step, reward)

