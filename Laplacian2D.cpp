static char help[] =
    "Solves the 2D Laplacian problem with KSP solver from PETSc\n \
Input parameters include : \n\
-view_exact_sol     : write exact solution vector to stdout \n\
-m <mesh_x>         : number of mesh points in x-direction \n\
-n <mesh_y>         : number of mesh points in y-direction\n\n";
#include <petscksp.h>

int main(int argc, char *argv[]) {

  Vec x, b, u; //approx solution, RHS, exact solution
  Mat A; //linear system matrix
  KSP ksp; //linear solver context
  PetscReal norm; //norm of solution error
  PetscInt i, j, Ii, J, Istart, Iend, m = 8, n = 7, its;
  PetscBool flg;
  PetscScalar v;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  //Compute the matrix and right-hand-side vector that define the linear system, Ax=b

  // Create parallel matrix, specifying only its global dimensions. When using
  // MatCreate(), the matrix format can be specified at runtime. also, the
  // parallel partitioning of the matrix is determined by PETSc at runtime.

  // Performance tuning note: For problems of substantial size, preallocation of
  // matrix memory is crucial for attaining good performance. See the matrix
  // chapter of the users manual for details.

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m * n, m * n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatMPIAIJSetPreallocation(A, 5, NULL, 5, NULL));
  PetscCall(MatSeqAIJSetPreallocation(A, 5, NULL));
  PetscCall(MatSeqSBAIJSetPreallocation(A, 1, 5, NULL));
  PetscCall(MatMPISBAIJSetPreallocation(A, 1, 5, NULL, 5, NULL));
  PetscCall(MatMPISELLSetPreallocation(A, 5, NULL, 5, NULL));
  PetscCall(MatSeqSELLSetPreallocation(A, 5, NULL));

  // Currently, all PETSc parallel matrix formats are partitioned by contiguous
  // chunks of rows across the processors. Determine which rows of the matrix
  // are locally owned
  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));

  // Set matrix elements for the 2-D Laplacian, five-point stencil in parallel.
  //-Each processor needs to insert only elements that it owns locally
  //(but  any non-local elements will be sent to the appropriate processor
  //during matrix assembly)
  //-Always specify global rows and colmns of matrix entries

  // Note: this uses the less common natural ordiring that orders first all the
  // unknowns for x=h then for x = 2h etc; Hence tou see J= Ii +- n instead of J
  // = I +- m as you might expect. The more standard ordering would first do all
  // variables for y=h, then y= 2h etc.;

  for (Ii = Istart; Ii < Iend; Ii++){
    v = -1.;
    i = Ii / n;
    j = Ii - i * n;
    if(i>0){
      J = Ii - n;
      PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, ADD_VALUES));
    }
    if(i < m-1){
      J = Ii + n;
      PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, ADD_VALUES));
    }
    if(j > 0){
      J = Ii - 1;
      PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, ADD_VALUES));
    }
    if (j < n - 1) {
      J = Ii + 1;
      PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, ADD_VALUES));
    }
    v = 4.;
    PetscCall(MatSetValues(A, 1, &Ii, 1, &Ii, &v, ADD_VALUES));
  }
  // Assemble matrix using the 2-step proecess:
  // MatAsselmblyBegin(), MatAssemblyEnd()
  // Computations can be done while messages are in transition by placing code
  // between these two statements.
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  //A is symmetric. Set symmetric flag to enable ICC/Cholesky preconditioner
  PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));
  // Create parallel vectors.
  //-We form 1 vector from scratch and then duplicate as needed.

  //-When using VecCreate(), VecSetSizes, and VecSetFromOptions() in this
  // examle, we specify only the vector's global dimensions; the parallel
  // partitioning is determined at runtime. 

  //-When solving a linear system, the
  //vectors and matrices MUST be partitioned accordingly. PETSc automatically
  //generates appropriately partitioned matrices and vectors when MatCreate()
  //and VecCreate() are used with the same communicator.

  //- The user can alternatively specify the local vector and matrix dimensions
  //when more sophisticated partitioning is needed (replacing PETSC_DECIDE
  //argment in the VecSetSizes() statement below)

  PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
  PetscCall(VecSetSizes(u, PETSC_DECIDE, m * n));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecDuplicate(u, &b));
  PetscCall(VecDuplicate(b, &x));

  // Set exact solution; then compute right-hand-side vector. By default we use
  // an exact solution of a vector with all elements of 1.0

  PetscCall(VecSet(u, 1.0));
  PetscCall(MatMult(A, u, b));

  //View the exact solution vector if desired
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-view_exact_sol", &flg, NULL));
  if(flg)
    PetscCall(VecView(u, PETSC_VIEWER_STDOUT_WORLD));

  // Create the linear solver;

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  //Set operators. Here the matrix that defines the linear system also serves
  //as the preconditioning matrix
  PetscCall(KSPSetOperators(ksp, A, A));

  // Set linear solver defaults for this problem (optional).
  //-By extracting the KSP and PC contexts form the KSP context,
  // we can then directly call any KSP and PC routines to set various options.
  //- The following statement is optional; all of these parameters could
  //alternativelly be specified at runtime via
  //KSPSetFromOptions(). All of these defaults can be overridden at runtime, as indicated below.

  PetscCall(KSPSetTolerances(ksp, 1.e-2 / ((m + 1) * (n + 1)), 1.e-50,
                             PETSC_CURRENT, PETSC_CURRENT));

  // set runtime options, e.g.,
  //    -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
  //    These options will override those specified above as long as
  //    KSPSetFromOptions() is call _after_ aby other customization
  //routines
  PetscCall(KSPSetFromOptions(ksp));
  //  Solve the linear system;
  PetscCall(KSPSolve(ksp, b, x));

  //Check the solution and clean up
  PetscCall(VecAXPY(x, -1.0, u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(KSPGetIterationNumber(ksp, &its));

  //Print convergence information. PetscPritf() produces a single
  //print statement from all processes that share a communicator.
  //An alternative is PetscFPrintf() which prints to a file.

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Norm of error %g iterations %" PetscInt_FMT "\n",
                        (double)norm, its));

  // Free the workspace. All PETSc objects should be destroyed.

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));

  //Always call PetscFinalize() before exiting : 
  //finalizes the PETSc libraries as well as MPI
  //Provides summary and diagnostic infformation if certain runtime
  //options are chosen (e.g., -log_view).
  PetscCall(PetscFinalize());

  return 0; 
  }