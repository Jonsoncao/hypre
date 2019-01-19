/*
 Example for Output Matrix with a grid for Hypre solver
 
 Interface:    Linear-Algebraic (IJ)
 
 Compile with: make exInputMatrix
 
 Sample run:   mpirun -np 2 exInputMatrix
 
 Description:  This example is modified from hypre Example 5 (IJ interface example). The linear system is imported from an input matrix and rhs. The solver is AMG.  */

#include <math.h>
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"

#include "vis.c"


int main (int argc, char *argv[])
{
    int myid, num_procs;
    int N, n;
    
    int ilower, iupper;
    int local_size, extra;
    
    int solver_id;
    int vis, print_system;
    
    HYPRE_IJMatrix A;
    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_IJVector b;
    HYPRE_ParVector par_b;
    HYPRE_IJVector x;
    HYPRE_ParVector par_x;
    
    HYPRE_Solver solver;
//    HYPRE_Solver precond;
    
    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    /* Default problem parameters */
    n = 33; // n is the row size for h = 1/32
    solver_id = 0;
    vis = 0;
    print_system = 0;
    
    
    /* Parse command line */
    {
        int arg_index = 0;
        int print_usage = 0;
        
        while (arg_index < argc)
        {
            if ( strcmp(argv[arg_index], "-n") == 0 )
            {
                arg_index++;
                n = atoi(argv[arg_index++]);
            }
            else if ( strcmp(argv[arg_index], "-solver") == 0 )
            {
                arg_index++;
                solver_id = atoi(argv[arg_index++]);
            }
            else if ( strcmp(argv[arg_index], "-vis") == 0 )
            {
                arg_index++;
                vis = 1;
            }
            else if ( strcmp(argv[arg_index], "-print_system") == 0 )
            {
                arg_index++;
                print_system = 1;
            }
            else if ( strcmp(argv[arg_index], "-help") == 0 )
            {
                print_usage = 1;
                break;
            }
            else
            {
                arg_index++;
            }
        }
        
        if ((print_usage) && (myid == 0))
        {
            printf("\n");
            printf("Usage: %s [<options>]\n", argv[0]);
            printf("\n");
            printf("  -n <n>              : problem size in each direction (default: 33)\n");
            printf("  -solver <ID>        : solver ID\n");
            printf("                        0  - AMG (default) \n");
            printf("                        1  - AMG-PCG\n");
            printf("                        8  - ParaSails-PCG\n");
            printf("                        50 - PCG\n");
            printf("                        61 - AMG-FlexGMRES\n");
            printf("  -vis                : save the solution for GLVis visualization\n");
            printf("  -print_system       : print the matrix and rhs\n");
            printf("\n");
        }
        
        if (print_usage)
        {
            MPI_Finalize();
            return (0);
        }
    }
    
    /* Preliminaries: want at least one processor per row */
    if (n*n < num_procs) n = sqrt(num_procs) + 1;
    N = n*n; /* global number of rows */
    
    /* Each processor knows only of its own rows - the range is denoted by ilower
     and upper.  Here we partition the rows. We account for the fact that
     N may not divide evenly by the number of processors. */
    local_size = N/num_procs;
    extra = N - local_size*num_procs;
    
    ilower = local_size*myid;
    ilower += hypre_min(myid, extra);
    
    iupper = local_size*(myid+1);
    iupper += hypre_min(myid+1, extra);
    iupper = iupper - 1;
    
    /* How many rows do I have? */
    local_size = iupper - ilower + 1;
    
//    /* Read the matrix from file */.
    
    HYPRE_IJMatrixRead("test_A", MPI_COMM_WORLD, HYPRE_PARCSR, &A);
    /* Get the parcsr matrix object to use */
    HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
    
    
    /* Read the rhs from and create solution vector*/
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&x);
    HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x);

    HYPRE_IJVectorRead( "test_rhs", MPI_COMM_WORLD,HYPRE_PARCSR, &b);
    HYPRE_IJVectorGetObject(b, (void **) &par_b);
    
    HYPRE_IJVectorAssemble(x);
    HYPRE_IJVectorGetObject(x, (void **) &par_x);
    
    
    /* solve the system */
    
    {
        int num_iterations;
        double final_res_norm;
        
        /* Create solver */
        HYPRE_BoomerAMGCreate(&solver);
        
        /* Set some parameters (See Reference Manual for more parameters) */
        HYPRE_BoomerAMGSetPrintLevel(solver, 3);  /* print solve info + parameters */
        HYPRE_BoomerAMGSetCoarsenType(solver, 6); /* Falgout coarsening */
        HYPRE_BoomerAMGSetRelaxType(solver, 3);   /* G-S/Jacobi hybrid relaxation */
        HYPRE_BoomerAMGSetNumSweeps(solver, 1);   /* Sweeeps on each level */
        HYPRE_BoomerAMGSetMaxLevels(solver, 20);  /* maximum number of levels */
        HYPRE_BoomerAMGSetTol(solver, 1e-7);      /* conv. tolerance */
        
        /* Now setup and solve! */
        HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x);
        HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);
        
        /* Run info - needed logging turned on */
        HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
        HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
        if (myid == 0)
        {
            printf("\n");
            printf("Iterations = %d\n", num_iterations);
            printf("Final Relative Residual Norm = %e\n", final_res_norm);
            printf("\n");
        }
        
        /* Destroy solver */
        HYPRE_BoomerAMGDestroy(solver);
    }
    
    /* Print the solution that can be exported back to Matlab */
    HYPRE_IJVectorPrint(x, "soln");
    
    /* Clean up */
    HYPRE_IJMatrixDestroy(A);
    HYPRE_IJVectorDestroy(b);
    HYPRE_IJVectorDestroy(x);
    
    /* Finalize MPI*/
    MPI_Finalize();
    
    return(0);
}

