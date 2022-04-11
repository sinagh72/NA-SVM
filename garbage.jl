include("src_new/BAN.jl")
include("src_new/ipqp.jl")
# include("src/Utils/src/createTable.jl")
# include("src/ArithmeticNonStandarNumbersLibrary/src/BAN.jl")
# include("src_old/ipqp.jl")
using .BAN
using StatsBase
using Distributions: Normal
using Random
using MLDataUtils
using MLJ
using DataFrames
using VegaLite
# using JLD2
# using LIBSVM
using Statistics
# using XLSX
# using Convex, SCS, ECOS
using LinearAlgebra

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
	round.(x_train, digits=8), round.(x_test, digits=8)
end

function fit(X, y, idx::Integer; M::Real=1.0,  
	C::Real=1.0, tol::Real=1e-8, maxit::Integer=30, verbose::Bool=false, genLatex::Bool=false, 
	kernel_params::Dict)

	n = length(X[:,1]); #size
	println(n);
	d = length(X[1,:]); #dimension

	c = convert(Vector{Ban}, [-ones(n).*η; zeros(n)]); #η multiplied to linear cost function!
	A = convert(Matrix{Ban}, [y' zeros(n)'; I(n) I(n)]);
	b = convert(Vector{Ban}, [0 ; C*ones(n)]);
	Q = convert(Matrix{Ban}, zeros(2n,2n));
	K = convert(Matrix{Ban}, zeros(n,n));
	
	for i ∈ 1 : n
		for j ∈ 1 : i
			t1 = _kernel(X[i,1:idx], X[j,1:idx]; params=kernel_params); #accessible
			t2 = _kernel(X[i,idx+1:d], X[j,idx+1:d]; params=kernel_params); #inaccessible 
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
	sol = ipqp(A,b,c,Q, tol; maxit=maxit, verbose=verbose, genLatex=genLatex, slack_var=n:2n+1);
	
	# lambdas = map(i -> standard_part(denoise(i,1e-4)), sol.x[1:n]);
	println("===============================")
	lambdas = sol.x[1:n]
	println.(sol.x[1:n])
	println("===============================")

	# lambdas = sol.x[1:n]
	# lambdas = sol.x[1:n];
	# lam_idx = findall(i -> tol < i <= C - tol , lambdas);
	lam_idx = findall(i -> 0 < i <= C - tol , lambdas);
	println.(lam_idx);
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
	if kernel_params["type"] == "linear"
		for i ∈ lam_idx
			w = w + lambdas[i]*y[i]*X[i,:]; 
		end
		w[1:idx] .*= η^(-1);
		# model.w = denoise(w, 1e-4);
	end
	# model.lambdas = sol.x[lam_idx]
	sol, lambdas, w, bb, lam_idx

end
function accuracy(x_test, y_test, x_train, y_train, lambdas, idx, b, kernel_params)
	n = length(x_test[:,1]);
	res = 0;
	for i ∈ 1:n
		y_pred = Ban(0);
			for j ∈ 1:length(y_train)
				y_pred += lambdas[j]*y_train[j]*(_kernel(x_train[j,1:idx], x_test[i,1:idx]; params=kernel_params) + 
				M*_kernel(x_train[j,idx+1:end], x_test[i,idx+1:end]; params=kernel_params)); 
			end
		y_pred += b;
		# dot(w,x_test) + b
		pred = sign(y_pred);
		if  pred == y_test[i]
			res += 1;
		else
			print("wrong prediction: ")
			print(",  Label: ");
			print(y_test[i]);
			print(",  predicted: ");
			println(pred);
		end
	end
	res /= n;
	res
end


#accessible feature index 1:idx
function testing(n_tc, n_x1, n_x2, x1, x2; folds=10, idx=2, C=10.0, normalize=false, standardize=false, tol=1e-8, split=0.7,
	kernel_params::Dict=("type"=>"linear", "π"=>1.0, "γ"=>1.0, "ξ"=>1.0, "κ"=>1.0, "σ"=>1.0))

	results = zeros(n_tc,6); #valid, # not converged, # lambdas < 2,scored/folds, score/valid, score
	dim = length(x1[1][1,1:end-1]);
	w = convert(Matrix{Ban}, zeros(n_tc,dim));
	b = convert(Vector{Ban}, zeros(n_tc));

	output = Vector{Matrix}(undef, n_tc);

	for j in 1:n_tc
		X = shuffleobs([x1[j];x2[j]], obsdim=1);
		# X = [x1[j];x2[j]];
		output[j] = X;
		y = X[:,end];
		X = X[:,1:end-1];
		train_idx = 0;
		val_idx = 0;
		itr = 0;
		if folds > 1
			train_idx, val_idx = kfolds(n_x1+n_x2, folds);
			itr = folds;
		else
			train_idx = [sample(1:n_x1+n_x2, round(Int,(n_x1+n_x2)*split), replace=false)];
			val_idx = [findall(i -> i ∉ train_idx, collect(1:n_x1+n_x2))];
			itr = 1;
		end
		score = 0;
		valid = 0;
		n_cons = 0;
		n_lambs = 0;
		for i ∈ 1:itr
			x_train, x_test = preprocess(X[train_idx[i],:], X[val_idx[i],:]; normalize=normalize, standardize=standardize);
			sol, lambdas, ww, bb, lam_idx = fit(x_train, y[train_idx[i]], idx; C=C, tol=tol, verbose=false, kernel_params=kernel_params);
			if sol.flag == false
				n_cons += 1;
				println()
				continue
			elseif length(lam_idx) < 2
				println();
				print("two λs not found");
				print(lam_idx);
				n_lambs += 1;
				continue
			end
			w[j,:] += ww;
			b[j] += bb;
			score += accuracy(x_test, y[val_idx[i]], x_train[lam_idx,:], y[train_idx[i]][lam_idx], lambdas[lam_idx], ww, bb, kernel_params);
			valid += 1;
			break
		end
		results[j,1] = valid;
		results[j,2] = n_cons;
		results[j,3] = n_lambs;
		results[j,4] = score/itr;
		results[j,5] = score/valid;
		results[j,6] = score;
		b[j] /= valid;
		w[j,:] ./= valid;
	end
	results, w, b, output
end

function testing_standard(t, C, n_x1, n_x2, input, idx; normalize=false, standardize=false,kernel=Kernel.Linear, γ=1.0, tol=1e-8)
	results = zeros(t,2);
	# w = zeros(t,dim);
	# b = zeros(t);

	for j in 1:t
		# X = shuffleobs([x1[j];x2[j]], obsdim=1);
		# X = [x1[j];x2[j]];

		y = input[j][:,end];
		X = input[j][:,1:idx];
		
		folds = 10;
		train_idx, val_idx = kfolds(n_x1+n_x2, folds);

		score = 0;
		valid = 0;
		for i ∈ 1:folds
			x_train, x_test = preprocess(X[train_idx[i],:], X[val_idx[i],:]; normalize=normalize, standardize=standardize);
			model = svmtrain(x_train', y[train_idx[i]]; svmtype=LIBSVM.SVC,kernel=kernel, gamma=γ, cost=C, tolerance=tol);
			ŷ, decision_values = svmpredict(model, x_test');
			score += mean(ŷ .== y[val_idx[i]]);
			valid += 1;
		end
		results[j,1] = valid;
		results[j,2] = score/valid;
		# b[j] /= valid;
		# w[j,:] ./= valid;

	end
	# results, w, b
	results
end
function standard_svm(t, C, n_x1, n_x2, x1, x2, idx; normalize=false, standardize=false)
	results = zeros(t,2);
	w = zeros(t,idx);
	b = zeros(t);
	for j in 1:t
		X = shuffleobs([x1[j];x2[j]], obsdim=1);
		y = X[:,end];
		X = X[:,1:idx];
		
		folds = 10;
		train_idx, val_idx = kfolds(n_x1+n_x2, folds);

		score = 0;
		valid = 0;
		for i ∈ 1:folds
			x_train, x_test = preprocess(X[train_idx[i],:], X[val_idx[i],:]; normalize=normalize, standardize=standardize);
			ww, bb = svm(x_train, y[train_idx[i]], C, idx);
			for test in x_test
				decision_value = dot(ww,test) + bb
				score += sign(decision_value) == y[val_idx[i]]
			end
			score /= length(x_test)
			valid += 1;
			w[j,:] += ww;
			b[j] += bb;
		end
		results[j,1] = valid;
		results[j,2] = score/valid;
		b[j] /= valid;
		w[j,:] ./= valid;
	end
	results, w, b
end

n_tc = 1; #number of test cases in a benchmark
x1 = Vector{Matrix}(undef, n_tc);
x2 = Vector{Matrix}(undef, n_tc);
n_x1 = 5; #number of samples in class 1 for each test case
n_x2 = 5; #number of samples in class 2 for each test case

#generating training data
for i ∈ 1:n_tc
	x1[i] = [sample_generator(n_x1, 1, 10, 2.5)  sample_generator(n_x1, 1, 10, 2.5) sample_generator(n_x1, 1, 0, 2.5) sample_generator(n_x1, 1, 0, 2.5) ones(n_x1)];
	x2[i] = [sample_generator(n_x2, 1, -10, 2.5) sample_generator(n_x2, 1,  -10, 2.5) sample_generator(n_x1, 1, 0, 2.5) sample_generator(n_x1, 1, 0, 2.5) -ones(n_x2)];
end

# save_object("data/linear_b3_20-4_x1.jld2",x1);
# save_object("data/linear_b3_20-4_x2.jld2",x2);
# x1 = load_object("data/linear_b1_20-4_x1.jld2");
# x2 = load_object("data/linear_b1_20-4_x2.jld2");
idx = 2; #end index of accessible features 
res, w, b, output = testing(n_tc, n_x1, n_x2, x1, x2; folds=1, idx=idx, C=10.0, normalize=false, standardize=true, tol=1e-8, 
kernel_params=Dict("type" => "linear", "π" => 1.0, "γ" => 1.0, "ξ" => 1.0, "κ" => 1.0, "σ" => 1.0));
res
w
b
b[isnan.(b)] .= Ban(0);
w[isnan.(w[:,1]),1] .= Ban(0);
w[isnan.(w[:,2]),2] .= Ban(0);
w[isnan.(w[:,3]),3] .= Ban(0);
w[isnan.(w[:,4]),4] .= Ban(0);

res2 = testing_standard(n_tc, 10.0, n_x1, n_x2, output, idx*2; normalize=false, standardize=true, tol=1e-8);
res2[:,2]
mean(res2[:,2])

XLSX.openxlsx("temp.xlsx", mode="w") do xf
    sheet = xf[1];
    # XLSX.rename!(sheet, "new_sheet");
    # sheet["A1"] = "this"
    # sheet["A2"] = "is a"
    # sheet["A3"] = "new file"
    # sheet["A4"] = 100

    # will add a row from "A5" to "E5"
    # will add a column from "B1" to "B4"
    sheet["A1", dim=1] = res[:,1];
    sheet["B1", dim=1] = res[:,2];
    sheet["C1", dim=1] = res[:,3];
    # sheet["A4", dim=1] = b
    # sheet["A5", dim=1] = w
    sheet["D1", dim=1] = res[:,4];
    sheet["E1", dim=1] = res[:,5];
    sheet["F1", dim=1] = res2[:,2];
	bS = replace.(string.(b), " + 0.0η^1" => "");
	bS = replace.(bS, " + 0.0η^2" => "");
    # bS = replace.(bS, " )" => ")");
    sheet["G1", dim=1] = bS;
    wS = replace.(string.(w[:,1]) .* string.(",  ") .* string.(w[:,2]) .* string.(",  ") .* string.(w[:,3]) .* string.(",  ") .* string.(w[:,4]), " + 0.0η^1" => "");
    wS = replace.(wS, " + 0.0η^2" => "");
    # wS = replace.(wS, " )" => ")");
    sheet["H1", dim=1] = wS;

end

x1[1][:,1:end-1]
x2[1][:,1:end-1]

x1[2][:,1:end-1]
x2[2][:,1:end-1]

x1[3][:,1:end-1]
x2[3][:,1:end-1]

x1[4][:,1:end-1]
x2[4][:,1:end-1]

x1[5][:,1:end-1]
x2[5][:,1:end-1]
# mean(w[:,1])
# mean(w[:,2])
# mean(w[:,3])
# mean(w[:,4])
# mean(b)

# save_object("data/linear_testcase1_5-2_x1.jld2",x1);
# save_object("data/linear_testcase1_5-2_x2.jld2",x2);



function testing(n_tc, n_x1, n_x2, x1, x2; folds=10, idx=2, C=10.0, normalize=false, standardize=false, tol=1e-8, split=0.7,
	kernel_params::Dict=("type"=>"linear", "π"=>1.0, "γ"=>1.0, "ξ"=>1.0, "κ"=>1.0, "σ"=>1.0))

	results = zeros(n_tc,6); #valid, # not converged, # lambdas < 2,scored/folds, score/valid, score
	dim = length(x1[1][1,1:end-1]);
	w = convert(Matrix{Ban}, zeros(n_tc,dim));
	b = convert(Vector{Ban}, zeros(n_tc));

	output = Vector{Matrix}(undef, n_tc);

	for j in 1:n_tc
		X = shuffleobs([x1[j];x2[j]], obsdim=1);
		# X = [x1[j];x2[j]];
		output[j] = X;
		y = X[:,end];
		X = X[:,1:end-1];
		train_idx = 0;
		val_idx = 0;
		itr = 0;
		if folds > 1
			train_idx, val_idx = kfolds(n_x1+n_x2, folds);
			itr = folds;
		else
			train_idx = [sample(1:n_x1+n_x2, round(Int,(n_x1+n_x2)*split), replace=false)];
			val_idx = [findall(i -> i ∉ train_idx, collect(1:n_x1+n_x2))];
			itr = 1;
		end
		score = 0;
		valid = 0;
		n_cons = 0;
		n_lambs = 0;
		for i ∈ 1:itr
			x_train, x_test = preprocess(X[train_idx[i],:], X[val_idx[i],:]; normalize=normalize, standardize=standardize);
			sol, lambdas, ww, bb, lam_idx = fit(x_train, y[train_idx[i]], idx; C=C, tol=tol, verbose=false, kernel_params=kernel_params);
			if sol.flag == false
				n_cons += 1;
				println()
				continue
			elseif length(lam_idx) < 2
				println();
				print("two λs not found");
				print(lam_idx);
				n_lambs += 1;
				continue
			end
			w[j,:] += ww;
			b[j] += bb;
			score += accuracy(x_test, y[val_idx[i]], x_train[lam_idx,:], y[train_idx[i]][lam_idx], lambdas[lam_idx], ww, bb, kernel_params);
			valid += 1;
			break
		end
		results[j,1] = valid;
		results[j,2] = n_cons;
		results[j,3] = n_lambs;
		results[j,4] = score/itr;
		results[j,5] = score/valid;
		results[j,6] = score;
		b[j] /= valid;
		w[j,:] ./= valid;
	end
	results, w, b, output
end

function testing_standard(t, C, n_x1, n_x2, input, idx; normalize=false, standardize=false,kernel=Kernel.Linear, γ=1.0, tol=1e-8)
	results = zeros(t,2);
	# w = zeros(t,dim);
	# b = zeros(t);

	for j in 1:t
		# X = shuffleobs([x1[j];x2[j]], obsdim=1);
		# X = [x1[j];x2[j]];

		y = input[j][:,end];
		X = input[j][:,1:idx];
		
		folds = 10;
		train_idx, val_idx = kfolds(n_x1+n_x2, folds);

		score = 0;
		valid = 0;
		for i ∈ 1:folds
			x_train, x_test = preprocess(X[train_idx[i],:], X[val_idx[i],:]; normalize=normalize, standardize=standardize);
			model = svmtrain(x_train', y[train_idx[i]]; svmtype=LIBSVM.SVC,kernel=kernel, gamma=γ, cost=C, tolerance=tol);
			ŷ, decision_values = svmpredict(model, x_test');
			score += mean(ŷ .== y[val_idx[i]]);
			valid += 1;
		end
		results[j,1] = valid;
		results[j,2] = score/valid;
		# b[j] /= valid;
		# w[j,:] ./= valid;

	end
	# results, w, b
	results
end

function pllot(x1, x2, l1, l2)
	fig = plot()
	Plots.scatter!(fig, x2[:,1],x2[:,2], annotationfontsize=2, mc=:red, markersize=5, label="class -1", series_annotations=Plots.text.(1:length(x1)+20, 8, :bottom))
	Plots.scatter!(fig, x1[:,1],x1[:,2], annotationfontsize=2, mc=:blue, markersize=5, label="class 1", series_annotations=Plots.text.(1:length(x1), 8, :bottom))
	Plots.xlabel!(l1);
	Plots.ylabel!(l2);
end
pllot(X[1:n_x1, 1:idx],X[n_x1+1:n_x1+n_x2,1:idx], "Feature 1", "Feature 2" )


x1[:,1]
String(X[1,1])
XLSX.openxlsx("temp.xlsx", mod43e="w") do xf
    sheet = xf[1];
    # XLSX.rename!(sheet, "new_sheet");
    # sheet["A1"] = "this"
    # sheet["A2"] = "is a"
    # sheet["A3"] = "new file"
    # sheet["A4"] = 100

    # will add a row from "A5" to "E5"
    # will add a column from "B1" to "B4"
    sheet["A1", dim=1] = res[:,1];
    sheet["B1", dim=1] = res[:,2];
    sheet["C1", dim=1] = res[:,3];
    # sheet["A4", dim=1] = b
    # sheet["A5", dim=1] = w
    sheet["D1", dim=1] = res[:,4];
    sheet["E1", dim=1] = res[:,5];
    sheet["F1", dim=1] = res2[:,2];
	bS = replace.(string.(b), " + 0.0η^1" => "");
	bS = replace.(bS, " + 0.0η^2" => "");
    # bS = replace.(bS, " )" => ")");
    sheet["G1", dim=1] = bS;
    wS = replace.(string.(w[:,1]) .* string.(",  ") .* string.(w[:,2]) .* string.(",  ") .* string.(w[:,3]) .* string.(",  ") .* string.(w[:,4]), " + 0.0η^1" => "");
    wS = replace.(wS, " + 0.0η^2" => "");
    # wS = replace.(wS, " )" => ")");
    sheet["H1", dim=1] = wS;

end
bb[isnan.(bb)] .= Ban(0);
	w[isnan.(w[:,1]),1] .= Ban(0);
	w[isnan.(w[:,2]),2] .= Ban(0);
	w[isnan.(w[:,3]),3] .= Ban(0);
	w[isnan.(w[:,4]),4] .= Ban(0);

# res, w, b, output = testing(n_tc, n_x1, n_x2, x1, x2; folds=1, idx=idx, C=10.0, normalize=false, standardize=true, tol=1e-8, 
# kernel_params=Dict("type" => "linear", "π" => 1.0, "γ" => 1.0, "ξ" => 1.0, "κ" => 1.0, "σ" => 1.0));
res
w
b


res2 = testing_standard(n_tc, 10.0, n_x1, n_x2, output, idx*2; normalize=false, standardize=true, tol=1e-8);
res2[:,2]
mean(res2[:,2])

x1[1][:,1:end-1]
x2[1][:,1:end-1]

x1[2][:,1:end-1]
x2[2][:,1:end-1]

x1[3][:,1:end-1]
x2[3][:,1:end-1]

x1[4][:,1:end-1]
x2[4][:,1:end-1]

x1[5][:,1:end-1]
x2[5][:,1:end-1]
# mean(w[:,1])
# mean(w[:,2])
# mean(w[:,3])
# mean(w[:,4])
# mean(b)

# save_object("data/linear_testcase1_5-2_x1.jld2",x1);
# save_object("data/linear_testcase1_5-2_x2.jld2",x2);


res = 0;
lam_idx = findall(i -> standard_part(i) > 1e-5 && i <= 1.0 , λ_ns);
lam_idx = λ_idx_ns
for i ∈ 1:n_x1+n_x2
	x_test = X[i, :];
	y_pred = Ban(0);
		for j ∈ 1:length(lam_idx)
			y_pred += λ_ns[lam_idx[j]]*Y[lam_idx[j]]*_kernel(X[lam_idx[j],:], x_test; params=kernel_params); 
		end
	y_pred += b_ns;
	println(y_pred)
	# dot(w,x_test) + b
	pred = sign(y_pred);
	if  pred == Y[i]
		res += 1;
	else
		print("wrong prediction: ")
		print(",  Label: ");
		print(Y[i]);
		print(",  predicted: ");
		println(pred);
	end
end
res