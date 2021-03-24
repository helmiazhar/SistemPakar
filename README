Helmi Azhar
1184013
D4 Teknik Informatika 3C
Sistem Pakar

def BFS(array):
    global tree2
    index = 0               # Merecord langkah-langkah yang diperlukan untuk mencapai node
    nodes_layers = [['S']]  # Merecord node yang dikunjungi
    solution = ['G']
    current_target = 'G'    # Node yang akan ditemukan dari akhir

    # Get visited nodes sequence
    while 'G' not in array:
        temp = []   # record nodes in each layer
        for item in tree2[array[index]]:
            if item in array:
                continue
            temp.append(item)
            array.append(item)
            if item == 'G':     # Akan berhenti jika tujuan sudah ditemukan
                break
        nodes_layers.append(temp)
        index += 1

    # Menemukan jalur yang optimal
    for i in range(index-1, 0, -1):
        for j in range(len(nodes_layers[i])):
            if current_target in tree2[nodes_layers[i][j]]:
                current_target = nodes_layers[i][j]
                solution.append(nodes_layers[i][j])
                break
    solution.append('S')    # Menambahkan node S ke awal solusi
    solution.reverse()      # Mengembalikan array solusi
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