# include("src/Utils/src/createTable.jl")
# include("src/ArithmeticNonStandarNumbersLibrary/src/BAN.jl")
# include("src_old/ipqp.jl")
include("src_new/BAN.jl")
include("src_new/ipqp.jl")
using .BAN
using StatsBase
using Distributions: Normal
using Random
using MLDataUtils
using MLJ
using DataFrames
using VegaLite
using JLD2
using LIBSVM
using Statistics
using XLSX
# using Convex, SCS, ECOS
using LinearAlgebra
using Plots

function _kernel(x1, x2; params::Dict=("type"=>"linear", "π"=>1.0, "γ"=>1.0, "ξ"=>1.0, "κ"=>1.0, "σ"=>1.0))
	kernel = params["type"];
	if kernel == "linear" #Linear
		return dot(x1,x2);
	elseif kernel == "polynomial" #Polynomial
		return (dot(x1,x2)+1)^params["π"];
	elseif kernel == "gaussian" #Gaussian
		return exp(-(norm(x1-x2)/2*params["σ"]^2));
	elseif kernel == "rbf" #RBF
		return exp(-params["γ"]*norm(x1-x2));
	elseif kernel == "laplace" #Laplace
		return exp(-(norm(x1-x2)/params["σ"]));
	elseif kernel == "hyperbolic" #Hyperbolic tangent   
		return tanh(params["κ"]*dot(x1,x2) + params["ξ"]);
	end
end

function sample_generator(n::Int, dim::Int, μ::Real=0, σ::Real=1)
	return rand(Normal(μ,σ), (n,dim))
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
	round.(x_train, digits=8), round.(x_test, digits=8), normalizer
end
#fitting NA-SVM
function fit(X, Y, idx::Integer; M::Ban=η,  
	C::Real=1.0, tol::Real=1e-8, maxit::Integer=30, verbose::Bool=false, genLatex::Bool=false, 
	kernel_params::Dict)

	n = length(X[:,1]); #size
	d = length(X[1,:]); #dimension

	c = convert(Vector{Ban}, [-ones(n).*M; zeros(n)]); #η multiplied to linear cost function!
	A = convert(Matrix{Ban}, [Y' zeros(n)'; I(n) I(n)]);
	b = convert(Vector{Ban}, [0 ; C*ones(n)]);
	Q = convert(Matrix{Ban}, zeros(2n,2n));
	K = convert(Matrix{Ban}, zeros(n,n));
	
	for i ∈ 1 : n
		for j ∈ 1 : i
			t1 = _kernel(X[i,1:idx], X[j,1:idx]; params=kernel_params); #accessible
			t2 = _kernel(X[i,idx+1:end], X[j,idx+1:end]; params=kernel_params); #inaccessible 
			K[i,j] = (t1 + t2*M);#scale down inaccessible in dual
			Q[i,j] = Y[i]*Y[j]*K[i,j];
			if i != j
				Q[i,j] /= 2;
				Q[j,i] = Q[i,j];
				K[i,j] /= 2;
				K[j,i] = K[i,j];
			end
		end
	end
	sol = ipqp(A,b,c,Q, tol; maxit=maxit, verbose=verbose, genLatex=genLatex, slack_var=n:2n+1);
	
	# lambdas = map(i -> standard_part(denoise(i,1e-6)), sol.x[1:n]);
	lambdas = sol.x[1:n];
	# lam_idx = findall(i -> tol < i <= C - tol , lambdas);
	lam_idx = findall(i -> tol*10 < i <= C , lambdas);
    bb = convert(Vector{Ban}, Y[lam_idx]);
	pos = findall(i-> i == Ban(1), bb);
	neg = findall(i-> i == Ban(-1), bb);
	for i = 1:length(lam_idx)
		for j ∈ lam_idx
			bb[i] -=  lambdas[j]*Y[j]*K[lam_idx[i],j];
		end
	end
	bb = (minimum(bb[pos]) + maximum(bb[neg]))/2;
	w =  convert(Vector{Ban},zeros(d));
	if kernel_params["type"] == "linear"
		w = ((lambdas[lam_idx].*Y[lam_idx])'*X[lam_idx,:])';
		w[idx+1:end] *= M;
	end
	sol, w, bb, lambdas, lam_idx, X, K, Q
end
#LIBSVM 
function standard_svm(X,Y; kernel=Kernel.Linear, tol=1e-8, C=1.0)
	model = svmtrain(X', Y; svmtype=LIBSVM.SVC,kernel=kernel, cost=C, tolerance=tol);
	model, model.SVs.X * model.coefs, -model.rho, model.coefs, model.SVs.indices  #w, b, λ, λ indices
end
#prediction non-standard
function pred_ns(λ_ns, λ_idx_ns, X_train, X_test, Y_train, Y_test, b, idx, kernel_params; M=Ban(η))
	res = 0;
	# lam_idx = findall(i -> standard_part(i) > 0 && i <= 1.0 , λ_ns);
	n = length(X_test[:,1]);
	y_res = convert(Vector{Ban}, zeros(n));
	for i ∈ 1:n
		x_test = X_test[i, :];
		y_pred = Ban(0);
			for j ∈ λ_idx_ns
				y_pred += λ_ns[j]*Y_train[j]*(_kernel(X_train[j,1:idx], x_test[1:idx]; params=kernel_params) + 
				M*_kernel(X_train[j,idx+1:end], x_test[idx+1:end]; params=kernel_params));
			end
		y_pred += b;
		y_res[i] = y_pred;
		pred = sign(y_pred);
		if  pred == Y_test[i]
			res += 1;
		else
			print("wrong prediction: ")
			print(",  Label: ");
			print(Y_test[i]);
			print(",  predicted: ");
			println(pred);
		end
	end
	res/n, y_res
end

n_tc = 1; #number of test cases in a benchmark
x1 = Vector{Matrix}(undef, n_tc);
x2 = Vector{Matrix}(undef, n_tc);
n_x1 = 30; #number of samples in class 1 for each test case
n_x2 = 30; #number of samples in class 2 for each test case
std_ = 3;
m = 2;
# generate training data
for i ∈ 1:n_tc
	x1[i] = [sample_generator(n_x1, 1, m*1, std_)  sample_generator(n_x1, 1, m*1, std_) sample_generator(n_x1, 1, m*0, std_)   sample_generator(n_x1, 1, m*0, std_) ones(n_x1)];
	x2[i] = [sample_generator(n_x2, 1, -m*1, std_)  sample_generator(n_x2, 1, -m*1, std_) sample_generator(n_x1, 1, -m*0, std_) sample_generator(n_x1, 1, -m*0, std_) -ones(n_x2)];
end
# save_object("train_data/linear_b6_30-4_x1.jld2",x1);
# save_object("train_data/linear_b6_30-4_x2.jld2",x2);
# x1 = load_object("train_data/linear_b8_20-4_x1.jld2");
# x2 = load_object("train_data/linear_b8_20-4_x2.jld2");
idx = 2; #last col index of accessible features
#data prepration
x1[1][:,1:end-1]
x2[1][:,1:end-1]
X = [x1[1];x2[1]]
Y = X[:,end]
X = X[:,1:end-1]

kernel_params = Dict("type" => "linear", "π" => 1.0, "γ" => 1.0, "ξ" => 1.0, "κ" => 1.0, "σ" => 1.0);
nrm = false;
x_train, _, normalizer = preprocess(X,X; normalize=nrm, standardize=!nrm); #normalization/standardization
model, w_s, b_s, λ_s, λ_idx_s = standard_svm(x_train, Y); #standard approach
sol, w_ns, b_ns, λ_ns, λ_idx_ns, X, K, Q = fit(x_train, Y, idx, maxit=25, kernel_params=kernel_params, M=Ban(η)); #non standard approach
#generating test data
n_test = 6;
x1_t = [sample_generator(n_test, 1, m*1, std_)  sample_generator(n_test, 1, m*1, std_) sample_generator(n_test, 1, m*0, std_) sample_generator(n_test, 1, m*0, std_) ones(n_test)];
x2_t = [sample_generator(n_test, 1, -m*1, std_) sample_generator(n_test, 1,  -m*1, std_) sample_generator(n_test, 1, -m*0, std_) sample_generator(n_test, 1, -m*0, std_) -ones(n_test)];

# save_object("test_data/linear_b6_6-4_x1.jld2",x1_t);
# save_object("test_data/linear_b6_6-4_x2.jld2",x2_t);
# x1_t = load_object("test_data/linear_b8_5-4_x1.jld2");
# x2_t = load_object("test_data/linear_b8_5-4_x2.jld2");
y_test = [x1_t[:,end];x2_t[:,end]]
X_test = [x1_t[:,1:end-1];x2_t[:,1:end-1]]
score = 0;
# x_test, _, _ = preprocess(X_test,X_test; normalize=nrm, standardize=!nrm)
x_test = StatsBase.transform(normalizer, X_test);
x_test
ŷ, decision_values = svmpredict(model, x_test');
score += mean(ŷ .== y_test)
ŷ
res, y_res = pred_ns(λ_ns, λ_idx_ns, X, x_test, Y, y_test, b_ns, 2, kernel_params);
y_res
res
sorted = sortslices([1:length(λ_ns) λ_ns],dims=1,by=x->x[2],rev=true);#sorting the SVs of NS approach
#storing the results
XLSX.openxlsx("temp.xlsx", mode="w") do xf
    sheet = xf[1];
    sheet["A1", dim=1] = w_s[:]

	wNS = replace.(string.(w_ns), " + 0.0η^1" => "");
	wNS = replace.(wNS, " + 0.0η^2" => "");
    sheet["B1", dim=1] = wNS[:]

	
    sheet["C1", dim=1] = b_s;
	bNS = replace.(string.(b_ns), " + 0.0η^1" => "");
	bNS = replace.(bNS, " + 0.0η^2" => "");
    sheet["D1"] = bNS;

	sheet["E1", dim=1] = string.(λ_idx_s);
	λ_temp_s = λ_s[21:end]*-1;
    sheet["F1", dim=1] = string.(abs.(λ_s)[:]);
    sheet["G1", dim=1] = Int.(standard_part.(sorted[:,1]))
	LNS = replace.(string.(sorted[:,2]), " + 0.0η^1" => "");
	LNS = replace.(LNS, " + 0.0η^2" => "");
    sheet["H1", dim=1] = LNS;
end

bD = zeros(length(λ_idx_s));
for j ∈ λ_idx_s
	counter = 1;
	bD[counter] = Y[j];
    for k ∈ λ_idx_s
		bD[counter] -= λ_s[counter]*dot(X_train[k,:],X_train[j,:]);
		counter += 1;
    end
end
bD = bD/length(λ_idx_s);


bD = zeros(length(λ_idx_s));
for j = 1:length(λ_idx_s)
	bD[j] = Y[λ_idx_s[j]];
	for k = 1:length(λ_idx_s)
		bD[j] -= λ_s[k]*dot(X_train[λ_idx_s[k],:],X_train[λ_idx_s[j],:]);
    end
end
bD
bD= median(sort(bD))
M=1;
res = 0;
wrong = 0;
for i = 1:length(x_test[:,1])
	tt = x_test[i,:];
	println(tt);
    y_pred = 0;
    for j = 1:length(λ_idx_s)
        y_pred = y_pred + λ_s[j]*dot(X_train[λ_idx_s[j],1:2], tt[1:2])+M*dot(X_train[λ_idx_s[j],3:4],tt[3:4]);
    end
	y_pred = y_pred-bD;
	pred = sign(y_pred);
	# println(y_pred);
	if  pred == y_test[i]
		res = res + 1;
    else
        wrong = wrong + 1;
    end
end
res
wrong

x_test


XLSX.openxlsx("temp.xlsx", mode="w") do xf
    sheet = xf[1];
    sheet["A1", dim=1] = x1[1][:,1];
    sheet["B1", dim=1] = x1[1][:,2];
    sheet["C1", dim=1] = x1[1][:,3];
    sheet["D1", dim=1] = x1[1][:,4];

	sheet["E1", dim=1] = x2[1][:,1];
    sheet["F1", dim=1] = x2[1][:,2];
    sheet["G1", dim=1] = x2[1][:,3];
    sheet["H1", dim=1] = x2[1][:,4];


	
end


