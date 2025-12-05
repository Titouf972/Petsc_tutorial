#include <petsc.h>

int main(int argc, char **argv) {
  const PetscInt N = 600000000;
  Vec bigVec;
  PetscInt i, istart, iend, local_size;
  PetscScalar *values = NULL;
  PetscInt *indices = NULL;
  PetscLogDouble t1, t2;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &bigVec));

  PetscCall(VecGetOwnershipRange(bigVec, &istart, &iend));
  PetscCall(PetscTime(&t1));
  local_size = iend - istart;
  indices = new PetscInt[local_size];
  values = new PetscScalar[local_size];
  for (int j = 0; j < local_size; j++){
    indices[j] = j + istart;
    values[j] = indices[j];
  }
  PetscCall(VecSetValues(bigVec, local_size, indices, values, INSERT_VALUES));
  delete[] indices;
  delete[] values;

  PetscCall(VecAssemblyBegin(bigVec));
  PetscCall(VecAssemblyEnd(bigVec));
  PetscCall(PetscTime(&t2));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Vector initialized in %lf seconds \n", t2 - t1));

  PetscCall(VecDestroy(&bigVec));
  PetscCall(PetscFinalize());
  return 0;
}