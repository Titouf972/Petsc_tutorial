#include <petsc.h>

int main (int argc, char** argv){
  const PetscInt nx = 10, ny = 10, stencil_size = 5;
  PetscInt i,  j, its;
  Mat A;
  Vec x, b;
  KSP solver;
  const PetscReal rtol = 1.e-8;
  KSPConvergedReason reason;
  PetscReal errorNorm, rnorm;
  PetscLogDouble t1, t2;
  DM dm;
  DMDALocalInfo info;
  const PetscInt stencilWidth = 1;
  MatStencil row, col5[stencil_size];
  PetscScalar hx2, hy2, coef, coef5[stencil_size];
  PetscScalar **bgrid;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  //Create the 2D DMDA object
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DMDA_STENCIL_STAR, nx, ny, PETSC_DECIDE, PETSC_DECIDE,
                         1, stencilWidth, NULL, NULL, &dm));

  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));

  //View the DMDA object
  PetscCall(DMView(dm, PETSC_VIEWER_STDOUT_WORLD));

  //Create the A matrix from the DMDA object
  PetscCall(DMCreateMatrix(dm, &A));

  //Retrieve local information from the DMA object
  PetscCall(DMDAGetLocalInfo(dm, &info));
  hx2 = 1. / ((info.mx - 1) * (info.mx - 1));
  hy2 = 1. / ((info.mx - 1) * (info.mx - 1));

  coef = 1.;
  coef5[0] = 2. / hx2 + 2. / hy2; //on peut avoir des pas différents en x et en y

  coef5[1] = -1. / hx2 ;
  coef5[2] = -1. / hx2;
  coef5[3] = -1. / hy2;
  coef5[4] = -1. / hy2;

  //Loop on the grid points
  for (j = info.ys; j < info.ys + info.ym; j++){
    for (i = info.xs; i < info.xs + info.xm; i++){
      row.i = i;
      row.j = j;
      row.c = 0;
      if(i == 0 || i== (info.mx - 1) || j == 0 || j == (info.my - 1)){
        //Set matrix values to enforece boundary conditions (homogeneous Dirichlet conditions)
        PetscCall(
            MatSetValuesStencil(A, 1, &row, 1, &row, &coef, INSERT_VALUES));
      }else{
        // Set matrix values of interior points
        col5[0].i = i;
        col5[0].j = j;
        col5[0].c = 0;

        col5[1].i = i-1;
        col5[1].j = j;
        col5[1].c = 0;

        col5[2].i = i+1;
        col5[2].j = j;
        col5[2].c = 0;

        col5[3].i = i;
        col5[3].j = j-1;
        col5[3].c = 0;

        col5[4].i = i;
        col5[4].j = j+1;
        col5[4].c = 0;
        PetscCall(MatSetValuesStencil(A, 1, &row, stencil_size, col5, coef5,
                                      INSERT_VALUES));
      }
    }
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  //Create gloval vectors b and x from DMDA object
  PetscCall(DMCreateGlobalVector(dm, &b));
  PetscCall(DMCreateGlobalVector(dm, &x));

  PetscCall(VecSet(b, 0.));

  //See dmda exercise to see how to put all the boundary to 1.0

  PetscCall(VecSetRandom(x, NULL));

  KSP linSolver;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &linSolver));
  PetscCall(KSPSetOperators(linSolver, A, A));
  PetscCall(KSPSetType(linSolver, KSPGMRES)); // Petsc uses GMRES by default
  PetscCall(KSPSetTolerances(linSolver, 1e-8, 1e-6, 100000,
                             10000)); // rtol, atol, maxdivits, maxits
  PetscCall(KSPSetInitialGuessNonzero(linSolver, PETSC_TRUE));

  PC MyPC;
  PetscCall(KSPGetPC(linSolver, &MyPC));

  // PC set Type should be done before the setFromOptions, the default Precond
  // is the PCBJACOBI
  PetscCall(PCSetType(MyPC, PCBJACOBI));

  // This will allow options for User
  PetscCall(KSPSetFromOptions(linSolver));
  PetscCall(PCSetUp(MyPC));
  PetscCall(KSPSetUp(linSolver));

  PetscCall(KSPView(linSolver, PETSC_VIEWER_STDOUT_WORLD));

  // Comparing exact solution and solution from solver

  // On résoud le système
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

  //PetscScalar rnorm;
  PetscCall(KSPGetResidualNorm(linSolver, &rnorm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Residual norm : %g \n", rnorm));

  PetscCall(KSPDestroy(&solver));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());

  return 0;
}