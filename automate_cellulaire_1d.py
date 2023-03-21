import numpy as np
import time
import matplotlib.pyplot as plt
import sys
from mpi4py import MPI

nombre_cas   : int = 256
nb_cellules  : int = 30  # Cellules fantomes
nb_iterations: int = 360

compute_time = 0.
display_time = 0.

def save_as_md(cells, symbols='⬜⬛'):
    res = np.empty(shape=cells.shape, dtype='<U')
    res[cells==0] = symbols[0]
    res[cells==1] = symbols[1]
    np.savetxt(f'resultat_{num_config:03d}.md', res, fmt='%s', delimiter='', header=f'Config #{num_config}', encoding='utf-8')

def save_as_png(cells):
    fig = plt.figure(figsize=(nb_iterations/10., nb_cellules/10.))
    ax = plt.axes()
    ax.set_axis_off()
    ax.imshow(cells[:, 1:-1], interpolation='none', cmap='RdPu')
    plt.savefig(f"resultat_{num_config:03d}.png", dpi=100, bbox_inches='tight')
    plt.close()

# On initialise MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# On divise le nombre de cas en 2
configs_per_process = nombre_cas // size
config_start = rank * configs_per_process
config_end = (rank + 1) * configs_per_process
if rank == size - 1:
    config_end = nombre_cas

# On passe dans la boucle entre le début et la fin (change pour chaque process)
for num_config in range(config_start, config_end):
    t1 = time.time()
    cells = np.zeros((nb_iterations, nb_cellules+2), dtype=np.int16)
    cells[0, (nb_cellules+2)//2] = 1
    for iter in range(1, nb_iterations):
        vals = np.left_shift(1, 4*cells[iter-1, 0:-2]
                             + 2*cells[iter-1, 1:-1]
                             + cells[iter-1, 2:])
        cells[iter, 1:-1] = np.logical_and(np.bitwise_and(vals, num_config), 1)
    t2 = time.time()
    compute_time += t2 - t1

    t1 = time.time()
    save_as_md(cells)
#    save_as_png(cells)
    t2 = time.time()
    display_time += t2 - t1

# On prend le compute time maximal entre les process, qui va nous donner le temps final (exécution parallèle)
compute_time_total = comm.reduce(compute_time, op=MPI.MAX, root=0)
display_time_total = comm.reduce(display_time, op=MPI.MAX, root=0)

# On affiche avec le process 0 les temps de display et de compute
if rank == 0:
    print(f"Temps calcul des generations de cellules : {compute_time_total:.2f} secondes")
    print(f"Temps d'affichage des resultats : {display_time_total:.2f} secondes")
