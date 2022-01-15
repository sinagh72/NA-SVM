include("./NA-SVM.jl")
using .NA_SVM

#generating samples
x1 = [0.25 0.25
    -0.25 0.25
    0.25 -0.25
     -0.25 -0.25];

x2 = [1 1
	 -1 -1
	 1 -1
	 -1 1];

#inaccessible features
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
model = NA_Svm();
model.C = 1/2;
model.γ = 1;
sol = NA_SVM.fit(model, X, y, idx);
model.w
NA_SVM.denoise(model.w,1e-4)
model.b

# model.x
#testing
#positive
X1 = [2.1 13];
X2 = [4.4 -55123];
X3 = [2.6 -123123];
X4 = [2.35 +123123];
X5 = [6 12];
X6 = [1.92 534];
X7 = [1.96 -534];
X8 = [1.256 -200];


NA_SVM.decision_value(model, X1)
NA_SVM.decision_value2(model, X1, idx)


NA_SVM.decision_value(model, X2)
NA_SVM.decision_value2(model, X2, idx)

NA_SVM.decision_value(model, X3)
NA_SVM.decision_value(model, X4)
NA_SVM.decision_value(model, X5)
NA_SVM.decision_value(model, X6)
NA_SVM.decision_value(model, X7)
NA_SVM.decision_value2(model, X7,idx)
NA_SVM.decision_value(model, X8)

#negative
X11 = [-6 13];
X12 = [-3.23 55123];
X13 = [0.4332 -123123];
X14 = [0.389 +123123];
X15 = [-1.28374 -122];
X16 = [0.1345 200];
X17 = [0.11 -200];

NA_SVM.decision_value(model, X11)
NA_SVM.decision_value(model, X12)
NA_SVM.decision_value(model, X13)
NA_SVM.decision_value(model, X14)
NA_SVM.decision_value(model, X15)
NA_SVM.decision_value(model, X16)
NA_SVM.decision_value(model, X17)

X21 = [0.5 2000];
X22 = [1 0];
X23 = [1 -11110];
X24 = [1 11110];

NA_SVM.decision_value(model, X21)
NA_SVM.decision_value(model, X22)
NA_SVM.decision_value(model, X23)
NA_SVM.decision_value(model, X24)

# using Distributions
# using Plots
# using BenchmarkTools


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

