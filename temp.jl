using LinearAlgebra: Matrix, norm2
using BenchmarkTools
using LinearAlgebra
using Distributions
using Plots
include("../../src/ArithmeticNonStandarNumbersLibrary/src/BAN.jl")
using .BAN
include("../../src_new/ipqp.jl")
# include("../../src_new/solve_standardqp_lex.jl")
# include("../../src_new/solve_standardqp.jl")
include("../../src/Utils/src/createTable.jl")


# x1 = [ 1.59405  4.2065
#  3.63758  3.26835
#  2.7821   5.10454
#  3.75954  4.04459];

#  x2 = [-0.578863   1.32849
#   0.776838   0.084364
#   0.857263   2.49895
#   0.0582129  1.88935];

# x1 = Float64[1 1
# 1 2
#   1 3
# 1 4];
# x2 = Float64[-1 -1
# -1 -2
# -1 -3
# -1 -4];



# x1 = [2.53901 3.44551
# 3.62893 3.38145
# 4.68039 2.70013
# 3.39256 3.71651
# 5.22046 3.60685
# 2.99639 2.02983
# 1.11872 4.62293];

# x2 = [-3.55557 -1.67534
# -2.26619 -2.55195
# -2.60944 -2.44757
# -3.40818 -0.532436
# -1.87657 -5.45636
# -2.73415 -1.41148
# -5.34419 -2.62808];

# Plots.scatter(x1[:,1],x1[:,2], color="blue", label="class1")
# Plots.xlabel!("x");
# Plots.ylabel!("y");
# Plots.scatter!(x2[:,1],x2[:,2], color="red", label="class2")


# t1 = Z[1:Int(n/2),:];
# t2 = Z[Int(n/2)+1:end,:];

# Plots.scatter(t1[:,1],t1[:,2], color="blue", label="class1")
# Plots.xlabel!("x");
# Plots.ylabel!("y");
# Plots.scatter!(t2[:,1],t2[:,2], color="red", label="class2")
# xx = -10:0.2:10;
# mD = (-w[1]/w[2])*xx .- (standard_part(b)/w[2]);
# sv1 = (-w[1]/w[2])*xx .+ (1-standard_part(b)/w[2]);
# sv2 = (-w[1]/w[	2])*xx .+ (-1-standard_part(b)/w[2]);
# Plots.plot!(xx,mD, lw=2, label="Margin")
# Plots.plot!(xx, sv1,lw=2, label="SV1")
# Plots.plot!(xx, sv2,lw=2, label="SV2")

# Plots.scatter!(x1_test[:,1],x1_test[:,2],color="green", label="x1_test")
# Plots.scatter!(x2_test[:,1],x2_test[:,2],color="pink", label="x2_test")

# test_size = 1;
# test_set = [sample_generator(test_size, 1) sample_generator(test_size, 1)];
# Plots.scatter!(x2_test[:,1],x2_test[:,2],color="black", label="x2_test")
# decision_value(test_set, X,y,lambdas,w, b)



function pllot(c1, c2, w, b)
	Plots.scatter(c1[:,1],c1[:,2], color="blue", label="class1")
	Plots.xlabel!("x");
	Plots.ylabel!("y");
	Plots.scatter!(c2[:,1],c2[:,2], color="red", label="class2")
	xx = 0:0.1:1;
	mD = standard_part(-w[1]/w[2]).*xx .- standard_part(b/w[2]);
	sv1 = standard_part(-w[1]/w[2]).*xx .+ standard_part((1-b)/w[2]);
	sv2 = standard_part(-w[1]/w[2]).*xx .+ standard_part((-1-b)/w[2]);
	# Plots.plot!(standard_part(mD) ,lw=2, label="Margin")
	# Plots.plot!(standard_part(sv1),lw=2, label="SV1")
	# Plots.plot!(standard_part(sv2),lw=2, label="SV2")
	mD, sv1, sv2
end

#####================================================================random sample
using Distributions: Normal
using MLDataUtils

function sample_generator(n::Int, dim::Int, μ::Float64=3.0, σ::Float64=1.0)
    rand(Normal(μ,σ), (n,dim))
end

#data
const data = [x1 ones(n1); x2 -ones(n2)];
train, test = splitobs(shuffleobs(data), at = 0.7, obsdim = 1);
x_train = train[:,1:dim];
y_train = train[:,dim+1:dim+1];
#####
#####================================================================random sample

function pllot(c1, c2, w, b)
	Plots.scatter(c1[:,1],c1[:,2], color="blue", label="class1")
	Plots.xlabel!("x");
	Plots.ylabel!("y");
	Plots.scatter!(c2[:,1],c2[:,2], color="red", label="class2")
	xx = 0:0.1:1;
	mD = standard_part(-w[1]/w[2]).*xx .- standard_part(b/w[2]);
	sv1 = standard_part(-w[1]/w[2]).*xx .+ standard_part((1-b)/w[2]);
	sv2 = standard_part(-w[1]/w[2]).*xx .+ standard_part((-1-b)/w[2]);
	# Plots.plot!(standard_part(mD) ,lw=2, label="Margin")
	# Plots.plot!(standard_part(sv1),lw=2, label="SV1")
	# Plots.plot!(standard_part(sv2),lw=2, label="SV2")
	mD, sv1, sv2
end

function NA_SVM(d1, d2, k_idx, C, tol, maxit, kernel; π::Float64=1.0, 
	γ::Float64=1.0, ξ::Float64=1.0, κ::Float64=1.0, σ::Float64=1.0, 
	verbose::Bool=true, genLatex::Bool=true)
	
	function k(x, y)
		if kernel == "l" #Linear
			return dot(x,y);
		elseif kernel == "p" #Polynomial
			return (dot(x,y)+1)^π;
		elseif kernel == "g" #Gaussian
			return exp(-(norm2(x-y)/2σ^2));
		elseif kernel == "rbf" #RBF
			return exp(-γ*norm2(x-y));
		elseif kernel == "la" #Laplace
			return exp(-(norm2(x-y)/σ));
		elseif kernel == "h" #Hyperbolic tangent   
			return tanh(κ*dot(x,y) + ξ);
		end
	end

	nd1 = length(d1[:,1]);
	nd2 = length(d2[:,1]);
	#total number of samples
	n = nd1 + nd2; 

	# m1 = [d1 ones(nd1)];
	# m2 = [d2 ones(nd2)];
	m1 = d1 ;
	m2 = d2 ;

	dim = length(m1[1,:]);
	#
	T = [m1 ; m2];
	# T = [d1 ; d2];
	y = [ones(nd1) ; -ones(nd2)];
	c = convert(Vector{Ban}, [-ones(n)*η ; zeros(n)]); #η multiplied to linear cost function!
	A = convert(Matrix{Ban}, [y' zeros(n)'; I(n) I(n)]);
	b = convert(Vector{Ban}, [0 ; C*ones(n)]);
	Q = convert(Matrix{Ban}, zeros(2n,2n));
	
	for i ∈ 1 : n
		for j ∈ 1 : i
			t1 = k(T[i,1:k_idx],T[j,1:k_idx]); #accessible
			t2 = k(T[i,k_idx+1:dim],T[j,k_idx+1:dim]); #inaccessible 
			Q[i,j] = y[i]*y[j]*(t1 + t2*η);#scale down inaccessible in dual
			if i != j
				Q[i,j]/2; #not sure yet!
				Q[j,i] = Q[i,j];
			end
		end
	end

	sol = ipqp(A,b,c,Q,tol; maxit=maxit, verbose=verbose, genLatex=genLatex, slack_var=n:2n+1);
	# sol = solve_standardqp(A,b,c,Q,tol, maxit; verbose=verbose, genLatex=genLatex, slack_var=n+1:2n);
	
	w = convert(Vector{Ban},zeros(dim));
	for i ∈ 1 : n
		w = w + sol.x[i]y[i]T[i,:]; 
	end
	# w[1:k_idx] .*= α;
	
	A, b, c, Q, n, T, y, sol, w

end

function predict(X)
	res = dot(X,w) - bb
	res
end

sample_size = 4;
feature_size = 2;

# sv(x, n, a, b) = [x+a:1:x+n+a x+b:1:x+n+b];
# g1,g2 = sample_generator(sample_size, feature_size) #generating random samples
# g1 = sv(0,sample_size, 1, 0);
# g2 = sv(0,sample_size, 0, 1);

g1 = [2 0 3 11
 2 1 4 12
 2 2 1 5
 2 3 6 -1];

g2 = [0 1 9 21
    0 2 42 12
    0 3 22 23
    0 4 6 3];


sample_size = length(g1[:,1]);
feature_size = length(g1[1,:]);
k_idx = 1; #for i ∈ 1:k_idx -> accessible features
C = 5;
tol = 1e-7;
maxit = 30;
verbose = true;
genLatex = false;

A, b, c, Q, n, T, y, sol, w = NA_SVM(g1,g2, k_idx, C, tol, maxit, "l"; verbose=verbose, genLatex=genLatex);

idx = findfirst(i -> tol < i < C - tol, sol.x[1:8]);
bb = dot(w,T[idx,:]);

X1 = [2 1000000]
X2 = [12 1]
X3 = [0 10]
X4 = [0 2231]
X5 = [5 13]

predict(X1)
predict(X2)
predict(X3)
predict(X4)
predict(X5)

# function NA_SVM2(d1, d2, a_index, C, tol, maxit, kernel; π::Float64=1.0, 
# 	γ::Float64=1.0, ξ::Float64=1.0, κ::Float64=1.0, σ::Float64=1.0, 
# 	verbose::Bool=true, genLatex::Bool=true)
	
# 	function k(x, y)
# 		if kernel == "l" #Linear
# 			return dot(x,y);
# 		elseif kernel == "p" #Polynomial
# 			return (dot(x,y)+1)^π;
# 		elseif kernel == "g" #Gaussian
# 			return exp(-(norm2(x-y)/2σ^2));
# 		elseif kernel == "rbf" #RBF
# 			return exp(-γ*norm2(x-y));
# 		elseif kernel == "la" #Laplace
# 			return exp(-(norm2(x-y)/σ));
# 		elseif kernel == "h" #Hyperbolic tangent   
# 			return tanh(κ*dot(x,y) + ξ);
# 		end
# 	end

# 	nd1 = length(d1[:,1]);
# 	nd2 = length(d2[:,1]);
# 	#total number of samples
# 	n = nd1 + nd2; 

# 	m1 = [d1 ones(nd1)];
# 	m2 = [d2 ones(nd2)];
# 	# m1 = d1 ;
# 	# m2 = d2 ;

# 	dim = length(m1[1,:]);
# 	#
# 	T = [m1 ; m2];
# 	# T = [d1 ; d2];
# 	y = [ones(nd1) ; -ones(nd2)];
# 	c = convert(Vector{Ban}, [-ones(n)*η ; zeros(n)]);
# 	A = convert(Matrix{Ban}, [y' zeros(n)'; I(n) I(n)]);
# 	b = convert(Vector{Ban}, [0 ; C*ones(n)]);
# 	Q = convert(Matrix{Ban}, zeros(2n,2n));
	
# 	for i ∈ 1 : n
# 		for j ∈ 1 : i
# 			t1 = k(T[i,1:a_index],T[j,1:a_index]);
# 			t2 = k(T[i,a_index+1:dim],T[j,a_index+1:dim]);
# 			Q[i,j] = y[i]*y[j]*(t1 + t2*η);
# 			if i != j
# 				Q[i,j] /= 2;
# 				Q[j,i] = Q[i,j];
# 			end
# 		end
# 	end

# 	sol = ipqp(A,b,c,Q,tol; maxit=maxit, verbose=verbose, genLatex=genLatex, slack_var=n+1:2n);

# 	w = zeros(dim,1);
# 	for i ∈ 1 : n
# 		w = w + sol.lam[i]*y[i]*T[i,:]; 
# 	end
# 	w[1:a_index] .*= α;
# 	idx = findfirst(i -> tol < i < C - tol, sol.lam);
# 	tmp = w'*T[idx,:];
# 	bb = convert(Ban,1/y[idx]) - tmp[1];


# 	A, b, c, Q, n, T, y, sol, w, bb


# end

# A2, b2, c2, Q2, n2, T2, y2, sol2, w2, b2 = NA_SVM2(g1,g2, a_index, C, tol, maxit, "l"; verbose=verbose, genLatex=genLatex);
# sol2.lam
# b2
# w2



# i = ind[1]
# b = y[i] - w'*T[i,:];
# md, sv1, sv2 = pllot(g1,g2,w,b)
# b
# w


#create a trivial data scatter
# compare the result of w of standard with the result of the non standard 
# why w in NS goes to infinity?
# one interpretation is that the matrix Q is degenerated -> calculate the eigen values





	# w = convert(Vector{Ban}, zeros(dim));
	# for i ∈ 1:n
	# 	w = w + sol.x[i]*y[i]*T[i,:];
	# end
	# ind = findall(i -> tol*10 < i < 1/(2n*λ) - tol*10 , sol.x[1:n]);
	# ind = findall(i -> tol*10 < i < 1/(2n*λ) - tol*10 , sol.x[1:n]);
	# i = ind[1];
	# b = y[i] - w'*T[i,:];
	# w, b, sol, y, T
	# sol = solve_standardqp(A,b,c,Q,tol, maxit; verbose=verbose, genLatex=genLatex, slack_var=n+1:2n);


	# if genLatex
	# 	println("");
	# 	println("");
	# 	println("");
	# 	preamble();
	# 	println("\t\\textbf{iter} & \$\\bm{r_1}\$ & \$\\bm{r_2}\$ & \$\\bm{r_3}\$ \\\\");
	# 	println("\t\\hline");
	# 	iter = 0;
	# 	for (r1, r2, r3) in eachrow(sol.r)
	# 		iter+=1;
	# 		print("\t $(iter) & \$"); print_latex(r1, digits=5, precision=64); 
	# 		print("\$ & \$");
	# 		print_latex(r2, digits=10, precision=64); print("\$ & \$"); 
	# 		print_latex(r3, digits=10, precision=64); println("\$ \\\\"); 
	# 		println("\t\\hline");
	# 	end
	# 	epilogue();
	# end
