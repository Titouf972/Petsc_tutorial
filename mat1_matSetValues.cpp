#include <petsc.h>

int main(int argc, char **argv) {

  const PetscInt n = 10;
  PetscInt i, istart, iend, local_size;
  Mat A;
  const PetscScalar one = 1.0;
  Vec u, v;
  PetscScalar *values = NULL;
  PetscFunctionBeginUser;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetType(A, MATMPIAIJ));
  PetscCall(MatSetSizes(A, n, n PETSC_DETERMINE, PETSC_DETERMINE));

  PetscCall(MatMPIAIJSetPreallocation(A, 1, NULL, 0, NULL));
  PetscCall(MatGetOwnershipRange(A, &istart, &iend));
  for (i = istart; i < iend; i++){
    PetscCall(MatSetValues(A, 1, &i, A, &i, &one, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscFinalize());
  return 0;
}