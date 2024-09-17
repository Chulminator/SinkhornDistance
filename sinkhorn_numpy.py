# Import necessary packages and functions
import numpy as np
import matplotlib.pyplot as plt

# Code is slightly changed from
# https://www.pragmatic.ml/sparse-sinkhorn-attention/


def sinkhorn_knopp(cost_matrix, source, target, reg):
    # Largest entries of P correspond to
    # movements with lowest cost
    eps=1e-4

    P = np.exp(-cost_matrix / reg)
    P /= P.sum()

    # Source corresponds to rows,
    # target corresponds to colums
    source = source.reshape(-1, 1)
    target = target.reshape(1, -1)

    err = 1
    ii = 0
    P_prev = np.copy(P)
    while err > eps:
        ii +=1
        # Over time this both the row_ratio and
        # col_ratio should approach vectors of all 1s
        # as our transport matrix approximation improves
        row_ratio = source / P.sum(axis=1, keepdims=True)
        P *= row_ratio
        col_ratio = target / P.sum(axis=0, keepdims=True)
        P *= col_ratio

        err = np.linalg.norm(P_prev-P,"fro")
        P_prev = np.copy(P)

    min_cost = np.sum(P * cost_matrix)
    return P, min_cost

source_dist = np.array([0.2, 0.3, 0.5])
target_dist = np.array([0.2, 0.35, 0.3, 0.15])
	    

cost_matrix = np.array([[3.1623, 7.0711, 7.0711, 6.3246 ],
                        [4, 7.2111, 6.3246, 5.831],
                        [5.3852, 7.2801, 5, 5 ]])

print("\ncost_matrix\n",cost_matrix)

transport_matrix, min_cost = sinkhorn_knopp(
    cost_matrix,
    source_dist,
    target_dist,
    reg=0.1
)

transport_matrix *= 2000
print("\nTransport matrix\n", transport_matrix)
print("\nMin cost\n", np.sum(transport_matrix * cost_matrix))

transport_matrix, min_cost = sinkhorn_knopp(
    cost_matrix,
    source_dist,
    target_dist,
    reg=0.5
)

transport_matrix *= 2000
print("\nTransport matrix\n", transport_matrix)
print("\nMin cost\n", np.sum(transport_matrix * cost_matrix))


transport_matrix, min_cost = sinkhorn_knopp(
    cost_matrix,
    source_dist,
    target_dist,
    reg=1.0
)

transport_matrix *= 2000
print("\nTransport matrix\n", transport_matrix)
print("\nMin cost\n", np.sum(transport_matrix * cost_matrix))
