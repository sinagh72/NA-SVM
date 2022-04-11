# unreduced form

function fact3(A,Q,x,s)
    m,n = size(A)

#           dλ        dx        ds
    M = [zeros(m,m)  A       zeros(m,n);
	     A'         -Q       I; #Matrix{Float64}(I,n,n);
	     zeros(n,m) diagm(s) diagm(x)]

    f = lu(M)

    #return M, f
    return f
end

function solve3(f,rb,rc,rxs)
    m = length(rb)
    n = length(rc)

    b = f\[rb; rc; rxs]

    dλ = b[1:m]
    dx = b[1+m:m+n]
    ds = b[1+m+n:m+2*n]

    return dλ,dx,ds
end
