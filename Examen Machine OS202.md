Examen Machine OS202
===
## Parallélisation MPI
### Question 1
On va ici séparer la boucle sur les nombre de cas entre les différents proccess pour pouvoir répartir le calcul mais aussi la sauvegarde entre les différents process. On a choisi une fragmentation statique car on va devoir effetuer le même nombre d'opération pour tous les threads.

```python=1
configs_per_process = nombre_cas // size
config_start = rank * configs_per_process
config_end = (rank + 1) * configs_per_process
if rank == size - 1:
    config_end = nombre_cas
```

### Question 2
![](https://markdown.data-ensta.fr/uploads/upload_5cd1fc3073acf699560975a15ddfdad8.png)

## Calcul d'une enveloppe
