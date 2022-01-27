module Run
	
include("./NA-SVM.jl")
using .NA_SVM
#generating samples
x1 = [2 1
	3 2
 	4 3
	5 4];
x2 = [2 -1
	3 -2
	4 -3
	5 -4];
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
model = NA_Svm(kernel="linear", tol=1e-8);
model.C = 1;
sol = NA_SVM.fit(model, X, y, idx);
println()
println("Value of b: ");
println(model.b);
println()
println("Value of w: ");
println(model.w);
println()
println("Lambdas (Before Denoising):")
println(sol.x[1:n])
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


# NA_SVM.decision_value(model, X8)
# NA_SVM.decision_value(model, X1)
# NA_SVM.decision_value(model, X2)
# NA_SVM.decision_value(model, X3)
# NA_SVM.decision_value(model, X4)
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