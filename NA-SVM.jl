__precompile__()
module NA_SVM

include("src/ArithmeticNonStandarNumbersLibrary/src/BAN.jl")
include("src_new/ipqp.jl")
include("src/Utils/src/createTable.jl")
using .BAN

export NA_Svm, fit, predict, decision_value, decision_value2

# NA-SVM declaration
mutable struct NA_Svm
    # Members
	b::Ban
	w::Vector{Ban}
	lambdas::Vector{Ban}
	y::Vector{Integer}
	x::Matrix{Ban}
	# sol::IpqpSolution
	idx::Integer
	C::Real
	tol::Real
	maxit::Integer
	kernel::String
	π::Float64
	γ::Float64
	ξ::Float64
	κ::Float64
	σ::Float64
	verbose::Bool
	genLatex::Bool
    # Constructor
	function NA_Svm(;C::Real=1.0, tol::Real=1e-7, maxit::Integer=30, kernel::String="linear", π::Float64=1.0, γ::Float64=1.0, ξ::Float64=1.0,
		κ::Float64=1.0, σ::Float64=1.0, verbose::Bool=true, genLatex::Bool=true)
	 	new(Ban(0), convert(Vector{Ban},zeros(1)), convert(Vector{Ban}, zeros(1)), zeros(1), convert(Matrix{Ban}, zeros(0,0)),
		 	0, C, tol, maxit, kernel, π, γ, ξ, κ, σ, verbose, genLatex)
	end
end

function _kernel(model::NA_Svm, x, y)
	if model.kernel == "linear" #Linear
		return dot(x,y);
	elseif model.kernel == "polynomial" #Polynomial
		return (dot(x,y)+1)^model.π;
	elseif model.kernel == "gaussina" #Gaussian
		return exp(-(norm(x-y)/2model.σ^2));
	elseif model.kernel == "rbf" #RBF
		return exp(-model.γ*norm(x-y));
	elseif model.kernel == "laplace" #Laplace
		return exp(-(norm(x-y)/model.σ));
	elseif model.kernel == "hyperbolic" #Hyperbolic tangent   
		return tanh(model.κ*dot(x,y) + model.ξ);
	end
end

function fit(model::NA_Svm, X, y, idx::Integer) 
	n = length(X[:,1]) #size
	d = length(X[1,:]) #dimension

	c = convert(Vector{Ban}, [-ones(n)*η ; zeros(n)]); #η multiplied to linear cost function!
	A = convert(Matrix{Ban}, [y' zeros(n)'; I(n) I(n)]);
	b = convert(Vector{Ban}, [0 ; model.C*ones(n)]);
	Q = convert(Matrix{Ban}, zeros(2n,2n));

	for i ∈ 1 : n
		for j ∈ 1 : i
			t1 = _kernel(model, X[i,1:idx],X[j,1:idx]); #accessible
			t2 = _kernel(model, X[i,idx+1:d],X[j,idx+1:d]); #inaccessible 
			Q[i,j] = y[i]*y[j]*(t1 + t2*η);#scale down inaccessible in dual
			if i != j
				Q[i,j]/2; #not sure yet!
				Q[j,i] = Q[i,j];
			end
		end
	end

	sol = ipqp(A,b,c,Q, model.tol; maxit=model.maxit, verbose=model.verbose, genLatex=model.genLatex, slack_var=n:2n+1);
	
	lam_idx = findall(i -> model.tol < i < model.C - model.tol, sol.x[1:n]);

	for i ∈ lam_idx
		model.b += y[i];
		t = [X[i,1:idx].*α X[i,idx+1:d]];
		for j ∈ lam_idx
			model.b -= sol.x[j]y[j]_kernel(model, t, X[j,:]);; 
		end
	end

	model.b /= length(lam_idx);

	if model.kernel == "linear"
		w = convert(Vector{Ban},zeros(d));
		for i ∈ 1 : n
			w = w + sol.x[i]y[i]X[i,:]; 
		end
	
		w[1:idx] .*= α;
		model.w = w;
		# model.w = denoise(w, 1e-4);
	end

	model.lambdas = sol.x[lam_idx]
	model.y = y[lam_idx]
	model.x = X[lam_idx,:];
	model.idx = idx
	sol
	###QUESTION
	#1. when should call function denoise?
end

function decision_value(model::NA_Svm, X)
	#make sure that X and model.w have the same dimensions 
	if model.kernel == "linear"
		return dot(model.w,X) + model.b;
	else
		y_pred = Ban(0);
		for i ∈ 1:length(model.y)
			t = [model.x[i,1:model.idx].*α model.x[i,model.idx+1:end]];
			y_pred += model.lambdas[i]model.y[i]_kernel(model, t, X);
		end
		y_pred += model.b
	end
end

function predict(model::NA_Svm, X)
	n = length(X[:,1]);
	y = zeros(n);
	for i ∈ 1: n
		y[i] = sign(decision_value(model,X[i,:]));
	end
	y
end

end