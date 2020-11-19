include("../HW1/arnoldi_val.jl")
include("../HW1/GS_val.jl")

function my_gmres_val(A, b, m, x_exact, tol)

    res_v = zeros(m,1);
    err_v = zeros(m,1);
    x = zeros(length(b),1);
    res = 1000;
    iter = 0;

    for i = 1:m
        e1 = zeros(i+1,1);
        e1[1] = 1;
        Q, H = arnoldi(A,b,i)
        b_norm = norm(b)
        z = H\(b_norm .*e1);
        x = Q[:,1:i]*z;
        res = norm(A*x-b)/norm(b);
        if res < tol
            break
        end
        res_v[i] = norm(A*x-b)/norm(b);	          # residual norm
        err_v[i] = norm(x-x_exact)/norm(x_exact); # error norm
    end
    return x, res_v, err_v
end
