# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:00:38 2021

@author: Asus
"""

tree = {'S': ['A', 'B', 'C'],
        'A': ['S', 'D', 'E', 'G'],
        'B': ['S', 'G'],
        'C': ['S', 'G'],
        'D': ['A'],
        'E': ['A']}

tree2 = {'S': ['A', 'B'],
         'A': ['S'],
         'B': ['S', 'C', 'D'],
         'C': ['B', 'E', 'F'],
         'D': ['B', 'G'],
         'E': ['C'],
         'F': ['C']
         }


def BFS(array):
    global tree2
    index = 0              
    nodes_layers = [['S']] 
    solution = ['G']
    current_target = 'G'    
    while 'G' not in array:
        temp = []  
        for item in tree2[array[index]]:
            if item in array:
                continue
            temp.append(item)
            array.append(item)
            if item == 'G':    
                break
        nodes_layers.append(temp)
        index += 1

    
    for i in range(index-1, 0, -1):
        for j in range(len(nodes_layers[i])):
            if current_target in tree2[nodes_layers[i][j]]:
                current_target = nodes_layers[i][j]
                solution.append(nodes_layers[i][j])
                break
    solution.append('S')   
    solution.reverse()      
    return solution, array


if __name__ == '__main__':
    solution, nodes_visited = BFS(['S'])
    print('Solusi Optimal: ' + str(solution))
    print('Node yang dikunjungi: ' + str(nodes_visited))
    
    

def DFS(array):
    global visited_list, tree
    if set(tree[array[-1]]).issubset(visited_list):
        del array[-1]
        return DFS(array)

    for item in tree[array[-1]]:
        if item in visited_list:
            continue
        visited_list.append(item)
        array.append(item)
        if item == 'G':
            return array, visited_list
        else:
            return DFS(array)


if __name__ == '__main__':
    visited_list = ['S']
    solution, visited_nodes = DFS(['S'])
    print('Solusi Optimal: ' + str(solution))
    print('Node yang dikunjungi: ' + str(visited_nodes))