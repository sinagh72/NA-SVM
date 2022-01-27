
include("src/ArithmeticNonStandarNumbersLibrary/src/BAN.jl")
include("src_new/ipqp.jl")
# include("src/Utils/src/createTable.jl")
using .BAN
using StatsBase
using Plots
using Distributions: Normal
using Random
using MLDataUtils
using MLJ
using DataFrames
using VegaLite
using JLD2

function _kernel(x, y; kernel::String="linear", π::Float64=1.0, γ::Float64=1.0, ξ::Float64=1.0,
    κ::Float64=1.0, σ::Float64=1.0)
	if kernel == "linear" #Linear
		return dot(x,y);
	elseif kernel == "polynomial" #Polynomial
		return (dot(x,y)+1)^π;
	elseif kernel == "gaussian" #Gaussian
		return exp(-(norm(x-y)/2*σ^2));
	elseif kernel == "rbf" #RBF
		return exp(-γ*norm(x-y));
	elseif kernel == "laplace" #Laplace
		return exp(-(norm(x-y)/σ));
	elseif kernel == "hyperbolic" #Hyperbolic tangent   
		return tanh(κ*dot(x,y) + ξ);
	end
end
function decision_value(x_test, X, y, lambdas, w, b, kernel, γ)
	#make sure that X and model.w have the same dimensions 
	# if kernel == "linear"
	# 	return dot(model.w,X) + model.b;
	# else
	y_pred = Ban(0);
	for i ∈ 1:length(y)
		y_pred += lambdas[i]*y[i]*_kernel(X[i,:], x_test; kernel=kernel, γ=γ); 
	end
	y_pred += b
	y_pred, dot(w,x_test) + b
	# end
end

function sample_generator(n::Int, dim::Int, μ::Real=0, σ::Real=1)
	return rand(Normal(μ,σ), (n,dim))
end
function shuffle_split(x1,x2, nx1, nx2, split::Float64=0.7)
	x1 = x1[shuffle(1:end), :];
	x2 = x2[shuffle(1:end), :];
	a_x1 = axes(x1, 1);
	a_x2 = axes(x2, 1);
	s_x1 = sample(a_x1, Int(floor(split*nx1)); replace=false);
	s_x2 = sample(a_x2, Int(floor(split*nx2)); replace=false);
	x1_train = x1[s_x1, :];
	x2_train = x2[s_x2, :];
	x1_test  = x1[[x for x ∈ a_x1 if x ∉ s_x1],:];
	x2_test  = x2[[x for x ∈ a_x2 if x ∉ s_x2],:];
	
	x1_train, x2_train, x1_test, x2_test
end

function accuracy(x_test, y_test, x_train, y_train, lambdas, w, b, kernel, γ)
	n = length(x_test[:,1]);
	res = 0;
	for i ∈ 1:n
		if sign((decision_value(x_test[i,:], x_train, y_train, lambdas, w, b, kernel, γ)[1])) == y_test[i]
			res += 1;
		else
			print("wrong prediction index: ")
			println(i);
		end
	end
	res /= n;
	res
end

function pplot(x1, x2)
	Plots.scatter(x1[:,1],x1[:,2], color="blue", label="class1")
	Plots.xlabel!("x");
	Plots.ylabel!("y");
	Plots.scatter!(x2[:,1],x2[:,2], color="red", label="class2")

end

function fit(X, y, idx::Integer; C::Real=1.0, tol::Real=1e-7, maxit::Integer=30, kernel::String="linear", 
    π::Float64=1.0, γ::Float64=1.0, ξ::Float64=1.0,
    κ::Float64=1.0, σ::Float64=1.0, verbose::Bool=false, genLatex::Bool=false)


	n = length(X[:,1]) #size
	d = length(X[1,:]) #dimension

	c = convert(Vector{Ban}, [-ones(n); zeros(n)]); #η multiplied to linear cost function!
	A = convert(Matrix{Ban}, [y' zeros(n)'; I(n) I(n)]);
	b = convert(Vector{Ban}, [0 ; C*ones(n)]);
	Q = convert(Matrix{Ban}, zeros(2n,2n));
	K = convert(Matrix{Ban}, zeros(n,n));
	
	for i ∈ 1 : n
		for j ∈ 1 : i
			t1 = _kernel(X[i,1:idx], X[j,1:idx]; kernel=kernel, γ=γ); #accessible
			t2 = _kernel(X[i,idx+1:d], X[j,idx+1:d]; kernel=kernel, γ=γ); #inaccessible 
			K[i,j] = (t1 + t2*η);#scale down inaccessible in dual
			Q[i,j] = y[i]*y[j]*K[i,j];
			if i != j
				# K[i,j] /= 2;
				K[j,i] = K[i,j];
				# Q[i,j] /= 2;
				Q[j,i] = Q[i,j];
			end
		end
	end

	sol = ipqp(A,b,c,Q, tol; maxit=maxit, verbose=false, genLatex=false, slack_var=n:2n+1);
	
	
	lambdas = map(i -> standard_part(denoise(i,1e-4)), sol.x[1:n]);
	lam_idx = findall(i -> tol < i <= C - tol , lambdas);
	

	# lam_idx = findall(i -> model.tol < i < model.C - model.tol, sol.x[1:n]);
    bb = Ban(0);
	for i ∈ lam_idx
		bb += y[i];
		# t = vec([X[i,1:idx].*η^(-1) X[i,idx+1:d]]);
		for j ∈ lam_idx
			bb -=  lambdas[j]*y[j]*K[i,j];
		end
	end
	length(lam_idx) > 0 ? bb = bb/length(lam_idx) : bb = Ban(0); 
	w =  convert(Vector{Ban},zeros(d));
	if kernel == "linear"
		for i ∈ lam_idx
			w = w + lambdas[i]*y[i]*X[i,:]; 
		end
		w[1:idx] .*= η^(-1);
		# model.w = denoise(w, 1e-4);
	end

	# model.lambdas = sol.x[lam_idx]
	sol, lambdas, w, bb, lam_idx, γ
	###QUESTION
	#1. exp(BAN) for non linear functions!

end

function preprocess(x_train, x_test; normalize::Bool=true, standardize::Bool=false)
	normalizer = Any;
	if normalize
		normalizer = StatsBase.fit(UnitRangeTransform, x_train, dims=1);
		x_train = StatsBase.transform(normalizer, x_train);
		x_test = StatsBase.transform(normalizer, x_test);
	elseif standardize
		normalizer = StatsBase.fit(ZScoreTransform, x_train, dims=1);
		x_train = StatsBase.transform(normalizer, x_train);
		x_test = StatsBase.transform(normalizer, x_test);
	end
	round.(x_train, digits=8), round.(x_test, digits=8)
end
#accessible feature index 1:idx
function testing(t, C, nx1, nx2, x1, x2; idx=1, normalize=false, standardize=false,kernel="linear", γ=1.0)
	results = zeros(t,6); #valid, # not converged, # lambdas < 2,scored/folds, score/valid, b, score
	dim = length(x1[1][1,1:end-1]);
	w = convert(Matrix{Ban}, zeros(t,dim));
	b = convert(Vector{Ban}, zeros(t));

	convert(Vector{Ban},zeros(t));
	for j in 1:t
		tol = 1e-8;

		X = shuffleobs([x1[j];x2[j]], obsdim=1);

		y = X[:,end];
		X = X[:,1:end-1];
		folds = 10;
		train_idx, val_idx = kfolds(nx1+nx2, folds);

		score = 0;
		valid = 0;
		n_cons = 0;
		n_lambs = 0;
		for i ∈ 1:folds
			x_train, x_test = preprocess(X[train_idx[i],:], X[val_idx[i],:]; normalize=normalize, standardize=standardize);
			sol, lambdas, ww, bb, lam_idx, γ =  fit(x_train, y[train_idx[i]], idx; 
			C=C, tol=tol, verbose=false, kernel=kernel, γ=γ);
			if sol.flag == false
				n_cons += 1;
				
				println()
				continue
			elseif length(lam_idx) < 2
				n_lambs += 1;
				continue
			end
			w[j,:] += ww;
			b[j] += bb;
			score += accuracy(x_test, y[val_idx[i]], x_train[lam_idx,:], y[train_idx[i]][lam_idx], lambdas[lam_idx], ww, bb, kernel, γ);
			valid += 1;
		end
		results[j,1] = valid;
		results[j,2] = n_cons;
		results[j,3] = n_lambs;
		results[j,4] = score/folds;
		results[j,5] = score/valid;
		results[j,6] = score;
		b[j] /= valid;
		w[j,:] ./= valid;

	end
	results, w, b
end


idx = 1;

t = 10;
x51 = Vector{Matrix}(undef, t);
x52 = Vector{Matrix}(undef, t);

x61 = Vector{Matrix}(undef, t);
x62 = Vector{Matrix}(undef, t);

nx1 = 5;
nx2 = 5;

x1 = load_object("linear_testcase4_10-2_x1.jld2");
x2 = load_object("linear_testcase4_10-2_x2.jld2");
for i ∈ 1:t
	# x1[i] = [sample_generator(nx1, 1, 2, 5) sample_generator(nx1, 1, 2, 5) ones(nx1)];
	# x2[i] = [sample_generator(nx1, 1, -3, 5) sample_generator(nx1, 1, -3, 5) -ones(nx2)];
	d1 = sample_generator(nx1, 1, 10, 3);
	d2 = sample_generator(nx1, 1, -10, 3);
	x61[i] = [x1[i][:,1:end-1]  d1 sample_generator(nx1, 1, 0, 5) x1[i][:,end]];
	x62[i] = [x2[i][:,1:end-1]  d2 sample_generator(nx2, 1, 0, 5) x2[i][:,end]];

	x51[i] = [x1[i][:,1:end-1] d1  x1[i][:,end]];
	x52[i] = [x2[i][:,1:end-1] d2 x2[i][:,end]];

end
nx1 = 10;
nx2 = 10;
# save_object("linear_testcase5_10-2_x1.jld2",x51);
# save_object("linear_testcase5_10-2_x2.jld2",x52);
# save_object("linear_testcase6_10-2_x1.jld2",x61);
# save_object("linear_testcase6_10-2_x2.jld2",x62);
x1 = load_object("linear_testcase4_10-2_x1.jld2");
x2 = load_object("linear_testcase4_10-2_x2.jld2");

res, w, b = testing(t, 10, nx1, nx2, x1, x2; idx=idx, normalize=true, standardize=false);
res
b[isnan.(b)] .= Ban(0);
w[isnan.(w[:,1]),1] .= Ban(0);
w[isnan.(w[:,2]),2] .= Ban(0);
# w[isnan.(w[:,3]),3] .= Ban(0);
# w[isnan.(w[:,4]),4] .= Ban(0);

w
b

mean(w[:,1])
mean(w[:,2])
# mean(w[:,3])
# mean(w[:,4])
mean(b)

  
m = mean(acc);
st = std(acc);
ci = 2*st/sqrt(t)
nx1 = 5;
nx2 = 5;

a = [1;2]
c = [2;3]
a+c

for i ∈ 1:t
	X, y = make_circles(nx1+nx2; noise=0.05, factor=0.3, as_table=false);
	y[y.==0] .= -1;
	x1[i] = [X[1:nx1,:] y[1:nx1]];
	x2[i] = [X[nx1+1:end,:] y[nx1+1:end]];
end

acc = run(t, 1, nx1, nx2, x1, x2, "rbf", 1.0, false, true);
m = mean(acc)
st = std(acc);
ci = 2*st/sqrt(t)
