# README #


### What is this repository for? ###

* This repository is part of a personal project to learn petsc and apply it to leverage applicatibility of my own project. In it I go throug the documentation

### How do I get set up? ###

* Clone directory
* navigate to petsc_tutorial directory
* cmake -B build
* cmake --build build
* executables are in the build directory

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### How to run ? ###

* mpirun -np 3 Laplacian2D -m 10 -n 10 -ksp_rtol 1.e-8 -ksp_monitor -ksp_type bcgs -pc_type asm

### Who do I talk to? ###

* Christophe-Alexandre Chalons