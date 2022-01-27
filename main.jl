include("./NA-SVM.jl")
using .NA_SVM
# using Distributions: Normal
# function sample_generator(n::Int, dim::Int, μ::Float64=3.0, σ::Float64=1.0)
    # rand(Normal(μ,σ), (n,dim))
# end

# x1 = sample_generator(4,2,3.0,1.0);
# x2 = sample_generator(4,2,1.0,1.0);
#generating samples
# x1 = Float64[2 1
# 3 2
#   4 3
# 5 4];
# x2 = Float64[2 -1
# 3 -2
# 4 -3
# 5 -4];
x1 = Float64[1 1
1 2
  1 3
1 4];
x2 = Float64[-1 -1
-1 -2
-1 -3
-1 -4];

# x1 = [ 1.59405  4.2065
#  3.63758  3.26835
#  2.7821   5.10454
#  3.75954  4.04459];

#  x2 = [-0.578863   1.32849
#   0.776838   0.084364
#   0.857263   2.49895
#   0.0582129  1.88935];
#from 1:idx are accessible features
idx = 1;
#feature size
dim = 2; 
#sample sizes
n1 = length(x1[:,1]);
n2 = length(x2[:,1]);
n = n1 + n2;
#
X = [x1; x2];
y = [ones(n1); -ones(n2)];
#model initiliazation
model = NA_Svm(kernel="linear", tol=1e-7);
model.C = 1;
sol = NA_SVM.fit(model, X, y, idx);

# map(i -> NA_SVM.standard_part(i), model.lambdas)
model.y
for i ∈ sol.x[1:n]
	println(NA_SVM.standard_part(NA_SVM.denoise(i,1e-4)));
	# println(NA_SVM.denoise(i,1e-4));
end
lam_idx = findall(i -> model.tol <i < model.C - model.tol, model.lambdas)

model.b = 0
for i ∈ lam_idx
	model.b += y[i]
	println(model.b)
	# t = vec([X[i,1:idx].*α X[i,idx+1:d]]);
	for j ∈ lam_idx
		println(y[j]*model.K[i,j])
		model.b -=  model.lambdas[j]*y[j]*model.K[i,j];
		println(model.b)

	end
	println(model.b)
end

model.b /= length(lam_idx);

println()
println("Value of b: ");
println(model.b);
println()
println("Value of w: ");
println(model.w);
println()
println()
println("Lambdas (After Denoising):")
println(NA_SVM.denoise(sol.x[1:n],1e-4)');

#testing
#positive

# X1 = [13 2];
# X2 = [-200 4.4];
# X3 = [+100 2.6];
# X4 = [+123123 2.35];
# X5 = [12 6];
# X6 = [534 1.92];
# X7 = [-534 1.96];
# X8 = [-121 1];

model.lambdas
NA_SVM.decision_value(model, [1   0])
NA_SVM.decision_value(model, [0 0])
NA_SVM.decision_value(model, [0 0.1])
NA_SVM.decision_value(model, [0 0])
NA_SVM.decision_value(model, [4 -1])

NA_SVM.decision_value(model, [0.563633 0.877861])
NA_SVM.decision_value(model, [0.224965 1.64325])
NA_SVM.decision_value(model, [0.868011 2.12276])
NA_SVM.decision_value(model, [1.58045 0.48299])


# NA_SVM.decision_value(model, X5)
# NA_SVM.decision_value(model, X6)
# NA_SVM.decision_value(model, X7)
# NA_SVM.decision_value(model, X8)

# #negative
# X11 = [13 -6 ];
# X12 = [55123 -3.23 ];
# X13 = [-123123 0.4332 ];
# X14 = [+123123 0.389 ];
# X15 = [-122 -1.28374 ];
# X16 = [200 0.1345 ];
# X17 = [-200 0.11 ];
# X18 = [2000 0.5 ];

# NA_SVM.decision_value(model, X11)
# NA_SVM.decision_value(model, X12)
# NA_SVM.decision_value(model, X13)
# NA_SVM.decision_value(model, X14)
# NA_SVM.decision_value(model, X15)
# NA_SVM.decision_value(model, X16)
# NA_SVM.decision_value(model, X17)
# NA_SVM.decision_value(model, X18)


# X22 = [1 0];
# X23 = [1 -11110];
# X24 = [1 11110];

# NA_SVM.decision_value(model, X22)
# NA_SVM.decision_value(model, X23)
# NA_SVM.decision_value(model, X24)

end