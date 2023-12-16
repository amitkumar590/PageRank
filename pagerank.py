import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

def read_graph(file_path):
    edges = np.loadtxt(file_path, skiprows=5, dtype=int)
    num_nodes = edges.max() + 1
    adjacency_matrix = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes))
    return adjacency_matrix

def pagerank_power_method(H, alpha, epsilon, max_iterations=1000):
    n = H.shape[0]
    
    # Initialize the PageRank vector
    pi = np.ones(n) / n
    
    # Construct the dangling node vector 'a'
    dangling_nodes = np.where(H.sum(axis=1) == 0)[0]
    a = sp.csr_matrix((np.ones(len(dangling_nodes)), (dangling_nodes, [0] * len(dangling_nodes))), shape=(n, 1))

    # Power method iteration
    k = 0
    residual = 1.0

    while residual >= epsilon and k < max_iterations:
        prev_pi = pi.copy()
        
        # Normalization after each iteration
        pi = alpha * (pi * H) + (alpha * (pi * a) + 1 - alpha) * (np.ones(n) / n)
        pi /= np.sum(pi)  # Normalize the PageRank vector
        
        residual = np.linalg.norm(pi - prev_pi, 1)
        
        k += 1

    if k == max_iterations:
        print("Warning: Maximum number of iterations reached without convergence.")

    return pi, k

# Applying Pagerank on Google Web Dataset - 2002
file_path = "web-Google.txt"
alpha = 0.85  
epsilon = 1e-8

# Read the graph and create the adjacency matrix
H = read_graph(file_path)

# Call the pagerank_power_method function
pagerank_vector, num_iterations = pagerank_power_method(H, alpha, epsilon)

# Printing number of Iterations
print("Number of Iteration : ",num_iterations)

# Get nodes and their PageRank values as tuples
nodes_and_values = list(enumerate(pagerank_vector, start=1))

# Sort the list in descending order of PageRank values
sorted_nodes_and_values = sorted(nodes_and_values, key=lambda x: x[1], reverse=True)

sorted_output_file = "unsorted_pagerank_output1.txt"
np.savetxt(sorted_output_file, nodes_and_values, fmt="%d %.18e", header=f"Node PageRank (For Google Dataset - 2002) ", comments="")
print("Results saved to", sorted_output_file)

# Save the sorted results to the output file
sorted_output_file = "sorted_pagerank_output1.txt"
np.savetxt(sorted_output_file, sorted_nodes_and_values, fmt="%d %.18e", header=f"Node PageRank (Sorted by PageRank value for Google DataSet - 2002)", comments="")
print("Sorted results saved to", sorted_output_file)

# Graphical Representation
top_n = 30 # Number of nodes to represent
top_nodes, top_values = zip(*sorted_nodes_and_values[:top_n])

# Create a scatter plot for better visualization
plt.figure(figsize=(10, 6))
plt.scatter(top_nodes, top_values, color='blue')
plt.xlabel('Node')
plt.ylabel('PageRank Value')
plt.title(f'Top {top_n} Nodes by PageRank Value')
plt.show()
