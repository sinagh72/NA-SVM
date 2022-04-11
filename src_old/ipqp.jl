using Printf
using SparseArrays
using LinearAlgebra
using Statistics

include("solver3.jl")
include("alpha_max.jl")
include("starting.jl")
include("solve_standardqp.jl")

struct IpqpSolution
    x::Vector # the solution vector 
    flag::Bool         # a true/false flag indicating convergence or not
    lam::Vector # the solution lambda in standard form
    s::Vector # the solution s in standard form
	r::Matrix # the residuals history vector
end

""" 
soln = ipqp(A,b,c,Q,tol) solves the quadratic program:

minimize c'*x + 0.5*x^T*Q*x where Ax = b x>=0

and the IpqpSolution contains fields 

[x,flag,lam,s,r]

which are interpreted as   
a flag indicating whether or not the
solution succeeded (flag = true => success and flag = false => failure),

along with the solution for the problem 

and the associated Lagrange multipliers (lam, s).

This solves the problem up to 
the duality measure (x'*s)/n <= tol and the normalized residual
norm([A'*lam + s - c; A*x - b; x.*s])/norm([b;c]) <= tol
and fails if this takes more than maxit iterations.
"""

function ipqp(A,b,c,Q,tol; maxit=100, verbose=false, genLatex=false, slack_var=[])
    
	###################
	# test input data #
	###################
    
	m0,n0 = size(A)
	mq,_ = size(Q)
    
    if mq != n0 || length(b) != m0 || length(c) != n0
        DimensionMismatch("Dimension of matrices A, b, c, Q mismatch. Check your input.")
    end
	
    ##############################
    # solve the original problem #
	##############################
    x,λ,s,flag,iter,r = solve_standardqp(A,b,c,Q,tol,maxit,verbose=verbose,genLatex=genLatex,slack_var=slack_var)

    if flag == true
        println("This problem has been solved in $(iter) iterations")
		print("Optimal value: "); println(0.5*(x'*Q*x)+dot(c, x));
		print("Optimal solution:"); println(x[setdiff(1:n0, slack_var)]);
	else
		print("The problem did not converge in $(maxit) iterations")
    end

    return IpqpSolution(vec(x),flag,vec(λ),vec(s),r)
end
