#include <petsc.h>

int main(int argc, char **argv ) {
  PetscMPIInt size, rank;
  Vec vec, vec2;
  PetscScalar product;
  PetscReal norm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &vec));
  PetscCall(VecSetType(vec, VECMPI));
  PetscCall(VecSetSizes(vec, rank + 1, PETSC_DETERMINE));
  PetscCall(VecSet(vec, size * 0.5));

  PetscCall(VecView(vec, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDuplicate(vec, &vec2));
  PetscCall(VecCopy(vec, vec2));

  PetscCall(VecDot(vec, vec2, &product));
  PetscCall(VecNorm(vec, NORM_2, &norm));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "[%d] dot product = %f | euclidean norm = %lf \n", rank,
                        product, norm, norm * norm));

  PetscCall(VecDestroy(&vec));
  PetscCall(VecDestroy(&vec2));

  PetscCall(PetscFinalize());

  return 0;
}