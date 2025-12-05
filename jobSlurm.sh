#!/bin/bash

#SBATCH --job-name=NOM_DU_JOB #nom du job
#SBATCH --nodes=<Number_of_nodes>
#SBATCH --ntasks=<Number_of_processors>
##SBATCH --gres=gpu:<Number> #nombre de gpu à réserver par noeud, ) décommenter sur paetition GPU
#SBATCH --hint=nomultithread # 1 processus MPI par coeur (pas d'hyperthreading)
#SBATCH --time=00:10:00 #Temps d'exécution maximum demnade (HH:MM:SS)
#SBATCH --output=nom_fichier_sortie.out
#SBATCH --error=nom_fichier_error.out

#On se place dans le répertoire de soumission
cd ${SLURM_SUBMIT_DIR}

# echo des commandes lancées

set -x

extra_args="-log_view -dm_mat_type aij -dm_vec_type mpi"
size=1000

#exécution du code
time srun ./dmda $extra_args -da_grid_x $size -da_grid_y $size -ksp_type cg -pc_type hypre