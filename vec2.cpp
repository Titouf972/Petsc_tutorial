#include <petsc.h>

int main(int argc, char **argv) { const PetscInt N = 600000000;
  Vec bigVec;
  PetscInt i, istart, iend, local_size;
  PetscScalar *values = NULL;
  PetscLogDouble t1, t2;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &bigVec));

  PetscCall(VecGetOwnershipRange(bigVec, &istart, &iend));
  PetscCall(PetscTime(&t1));
  local_size = iend - istart;
  PetscCall(VecGetArray(bigVec, &values));
  for (i = 0; i < local_size; i++) {
    values[i] = static_cast<PetscScalar>(i + istart);
  }
  PetscCall(VecRestoreArray(bigVec, &values));
  PetscCall(PetscTime(&t2));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Vector initialized in %lf seconds \n", t2 - t1));

  PetscCall(VecDestroy(&bigVec));
  PetscCall(PetscFinalize());
  return 0;
}