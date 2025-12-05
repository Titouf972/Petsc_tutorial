#include <petsc.h>

int main(int argc, char **argv) {
  const PetscInt N = 600000000;
  Vec bigVec;
  PetscInt i, istart, iend;
  PetscScalar *values = NULL;
  PetscInt *indices = NULL;
  PetscLogDouble t1, t2;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &bigVec));

  PetscCall(VecGetOwnershipRange(bigVec, &istart, &iend));
  PetscCall(PetscTime(&t1));
  for (int j = istart; j < iend; j++) {
    PetscCall(VecSetValue(bigVec, i, (PetscScalar)i, INSERT_VALUES));
  }

  PetscCall(VecAssemblyBegin(bigVec));
  PetscCall(VecAssemblyEnd(bigVec));
  PetscCall(PetscTime(&t2));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Vector initialized in %lf seconds \n", t2 - t1));

  PetscCall(VecDestroy(&bigVec));
  PetscCall(PetscFinalize());
  return 0;
}