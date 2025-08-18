static char help[] = "Solves a tridiagonal linear system with KSP \n\n";
#include <petscksp.h>
/*
'petscksp.h automatically includes the following :
petscsys.h - base petsc routines
petscmat.h - matrices
petscis.h - index sets
petscviewer.h - viewers
petscvec.h - vectors
petscpc.h -preconditioners

Note: for corresponding parallel example go to ex23
*/
#include <iostream>

int main(int argc, char *argv[]) {

  Vec x, b, u; // approx solution, RHS, exact solution
  Mat A; //Linear system matrix
  KSP ksp; //linear solver context
  PC pc; //Preconditioner context
  PetscReal norm; //Norm of solution error
  PetscInt i, n = 10, col[3], its;
  PetscMPIInt size;
  PetscScalar value[3];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE,
             "This is a uniprocessor example only");

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  // Now compute the matrix and the right-hand side vector that define the
  // linear system, Ax=b;

  // Create vectors.Note that we form 1 vector from scratch and then duplicate
  // as needed;

  PetscCall(VecCreate(PETSC_COMM_SELF, &x));
  PetscCall(PetscObjectSetName((PetscObject)x, "Solution"));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(VecDuplicate(x, &u));

  //Create the matrix. When using MatCreate(), the matrix format can be specified at runtime.

  // Perfomance tuning note: For problems of substantial size, preallocation of
  // matrix memory is crucial for attaining good perfomance. See the matrix
  // chapter of the ussers manual for details.;

  PetscCall(MatCreate(PETSC_COMM_SELF, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  //Assemble Matrix

  value[0] = -1.;
  value[1] = 2.;
  value[2] = -1.;
  for (i = 1; i < n - 1; i++) {
    col[0] = i - 1;
    col[1] = i;
    col[2] = i + 1;
    PetscCall(MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES));
  }
  if(n > 1){
    i = n - 1;
    col[0] = n - 2;
    col[1] = n - 1;
    PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
  }
  i = 0;
  col[0] = 0;
  col[1] = 1;
  value[0] = 2.;
  value[1] = -1.;

  PetscCall(MatSetValues(A, 1, &i, n > 1 ? 2 : 1, col, value, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  // Set exact solution; then compute right-hand-side vector.

  PetscCall(VecSet(u, 1.));
  PetscCall(MatMult(A, u, b));

  //Create the linear solver and set various options

  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
  // Set operators.Here the matrix that defines the linear system also serves as
  // the matrix that defines the preconditioner;

  PetscCall(KSPSetOperators(ksp, A, A));

  // Set linear solver defaults for this problem (optional).
  // By extracting the ksp and PC contexts form the KSP context, we can then
  // directly call any KSP and PC routines to set various options;

  // The following four statements are optional; all of these parameters could
  // alternatively be specified at runtime via KSPSetFromOptions;
  if(!PCMPIServerActive){ //Cannot directly set KSP/PC options when using the MPI linear solver
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCJACOBI));
    PetscCall(KSPSetTolerances(ksp, 1.e-5, PETSC_CURRENT, PETSC_CURRENT,
                               PETSC_CURRENT));
  }
  // Set runtime options, e.g.,
  //-ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
  // These options will override thise specified above as long as
  // KSPSetFromOptions() is called _after_ any other customization routines;

  PetscCall(KSPSetFromOptions(ksp));

  //Solve the linear system

  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(VecView(b, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatView(A, NULL));

  //Check the solution and clean up

  PetscCall(VecAXPY(x, -1., u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
                        "Norm of error %g, Iterations %" PetscInt_FMT "\n",
                        (double)norm, its));
  // Check that KPS automatically handles the fact that the new non-zero values
  // in the matrix are propagated to the KSP Solver;
  PetscCall(MatShift(A, 2.0));
  PetscCall(KSPSolve(ksp, b, x));

  // Free work space. All PETSc objects should be destroyed when they areno
  // longer needed.;

  PetscCall(KSPDestroy(&ksp));
  //test if prefixes properly proagate to PCMPI objects
  if(PCMPIServerActive){
    PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
    PetscCall(KSPSetOptionsPrefix(ksp, "prefix_test_"));
    PetscCall(MatSetOptionsPrefix(A, "prefix_test_"));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(KSPDestroy(&ksp));
  }

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));

  // Always call PetscFinalize() before exiting a program. This routine
  //-Finalizes the PETSc libraries as well as MPI
  //-provides summary and diagnostic information if certain runtime options are
  //chosen (e.g. -log_view);

  PetscCall(PetscFinalize());

  return 0;
}
