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
  //Using MatCreateConstantDiagonal: C'est la matrice identité
  PetscCall(MatCreateConstantDiagonal(PETSC_COMM_WORLD, n, n, PETSC_DETERMINE,
                                      PETSC_DETERMINE, 1., &A));
  //On scale la matrice identité
  PetscCall(MatScale(A, 0.1));

  //On rend le découpage des vecteurs u et v compatibles avec le découpage de la matrice A
  PetscCall(MatCreateVecs(A, &u, &v));

  //On initialise u
  PetscCall(VecGetOwnershipRange(u, &istart, &iend));
  local_size = iend - istart;

  PetscCall(VecGetArray(u, &values));
  for (i = 0; i < local_size; i++){
    values[i] = static_cast<PetscScalar>(i + istart + 1) * 10;
  }
  PetscCall(VecRestoreArray(u, &values));

  //On multiplie A par u et on stocke le résultat dans v
  PetscCall(MatMult(A, u, v));

  PetscCall(VecView(v, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&v));

  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());

  return 0;
}