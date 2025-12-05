#include <petsc.h>

int main(int argc, char **argv) {

PetscInt NpointsPetDir = 10, N = NpointsPetDir * NpointsPetDir,
                 stencil_size = 5;
  PetscInt i, istart, iend, colIndex, rowIndex;
  //const PetscScalar h = 1. / (NpointsPetDir + 1);
  const PetscScalar h = 1.;
  Mat A;
  PetscInt indexes[stencil_size];
  PetscScalar values[] = {-1. / (h * h), -1. / (h * h), 4. / (h * h),
                          -1. / (h * h), -1. / (h * h)};
  PetscLogDouble t1, t2;
  PetscLogStage stage1, stage2;
  KSPConvergedReason reason;
  PetscInt its;
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  //Allow to put size as an option
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-size", &NpointsPetDir, NULL));
  PetscCall(
      PetscPrintf(PETSC_COMM_WORLD, "NpointsPerDir %d \n", NpointsPetDir));

//Recalculte N
  N = NpointsPetDir * NpointsPetDir;

  //Use stages
  PetscCall(PetscLogStageRegister("Fill and assemble", &stage1));
  PetscCall(PetscLogStageRegister("Solve", &stage2));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetType(A, MATMPIAIJ));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(
      MatMPIAIJSetPreallocation(A, stencil_size, NULL, stencil_size - 1, NULL));

  //PetscCall(MatMPIAIJSetPreallocation(A, 0, NULL, 0, NULL));
  //PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  PetscCall(MatGetOwnershipRange(A, &istart, &iend));
  PetscCall(PetscLogStagePush(stage1));
  for (i = istart; i < iend; i++) {
    rowIndex = i / NpointsPetDir;
    colIndex = i - rowIndex * NpointsPetDir;

    //Negative indexes are ignored
    indexes[0] = (rowIndex > 0) ? (i - NpointsPetDir) : -1;
    indexes[1] = (colIndex > 0) ? (i - 1) : -1;
    indexes[2] = i;
    indexes[3] = (colIndex < NpointsPetDir - 1) ? (i + 1) : -1;
    indexes[4] = (rowIndex < NpointsPetDir - 1) ? (i + NpointsPetDir) : -1;

    PetscCall(
        MatSetValues(A, 1, &i, stencil_size, indexes, values, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscLogStagePop());

  //PetscCall(MatView(A, PETSC_VIEWER_DRAW_WORLD));
  //PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  KSP linSolver;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &linSolver));
  PetscCall(KSPSetOperators(linSolver, A, A));
  PetscCall(KSPSetType(linSolver, KSPGMRES)); //Petsc uses GMRES by default
  PetscCall(KSPSetTolerances(linSolver, 1e-8, 1e-6, 100000, 10000)); //rtol, atol, maxdivits, maxits

  PC MyPC;
  PetscCall(KSPGetPC(linSolver, &MyPC));

  //PC set Type should be done before the setFromOptions, the default Precond is the PCBJACOBI
  PetscCall(PCSetType(MyPC, PCBJACOBI));

  //This will allow options for User
  PetscCall(KSPSetFromOptions(linSolver));
  PetscCall(PCSetUp(MyPC));

  //Comparing exact solution and solution from solver

  Vec x_exact, b;
  //On initialise x_exact et b ici
  PetscCall(MatCreateVecs(A, &x_exact, &b));
  PetscCall(VecSetRandom(x_exact, NULL));

  PetscCall(MatMult(A, x_exact, b));

  Vec x;
  PetscCall(VecDuplicate(x_exact, &x));

  //On résoud le système
  PetscCall(PetscLogStagePush(stage2));
  PetscCall(PetscTime(&t1));
  PetscCall(KSPSolve(linSolver, b, x));
  PetscCall(PetscTime(&t2));
  PetscCall(PetscLogStagePop());

  PetscCall(
      PetscPrintf(PETSC_COMM_WORLD, "Elapse time : %lf seconds \n", t2 - t1));

  PetscCall(KSPGetConvergedReason(linSolver, &reason));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Converged reason : %s \n",
                        KSPConvergedReasons[reason]));

  PetscCall(KSPGetIterationNumber(linSolver, &its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Iterations number : %d \n", its));

  PetscScalar rnorm;
  PetscCall(KSPGetResidualNorm(linSolver, &rnorm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Residual norm : %g \n", rnorm));

  Vec x_diff;
  PetscCall(VecDuplicate(x_exact, &x_diff));
  PetscCall(VecCopy(x_exact, x_diff));

  PetscScalar alpha;
  alpha = -1.;
  PetscCall(VecAXPY(x_diff, alpha, x));

  PetscScalar norm2;
  PetscScalar norm_inf;

  PetscCall(VecNorm(x_diff, NORM_2, &norm2));
  PetscCall(VecNorm(x_diff, NORM_INFINITY, &norm_inf));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "My norm 2 is %lf \n", norm2));
  PetscCall(
      PetscPrintf(PETSC_COMM_WORLD, "My norm infinity is %lf \n", norm_inf));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&x_exact));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x_diff));
  PetscCall(PetscFinalize());

  //ksp_norm_type unpreconditioned pour la norm non-préconditionnée.
    //ksp_type, pc_type
  return 0;
}