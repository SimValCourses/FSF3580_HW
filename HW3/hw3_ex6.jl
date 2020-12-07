using LinearAlgebra, PyPlot

f = z->exp(π).*[[1 (exp(z)-1)/z];[0 exp(z)]]

epsv = 10.0.^collect(-18:-1)
errv = zeros(length(epsv))
for (i,ϵ) in enumerate(epsv)
    println()
    println("ϵ = $ϵ")
    A = [[π 1];[0 π+ϵ]]
    X = eigvecs(A);
    display(X)
    display(inv(X))
    D = eigvals(A);
    println(D)
    println(D[1]-D[2])
    # Jordan definition
    F = X*diagm(exp.(D))*inv(X)
    # exact calculation of exp(A)
    Fe = f(ϵ)
    # comparison
    errv[i] = norm(Fe-F)
end

figure(1)
loglog(epsv,errv)
