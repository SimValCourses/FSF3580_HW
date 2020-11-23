include("../HW1/arnoldi_val.jl")
include("../HW1/GS_val.jl")

function my_gmres_val(A, b, m, x_exact, tol)

    res_v = zeros(m,1);
    err_v = zeros(m,1);
    time_v = zeros(m,1);
    x = zeros(length(b),1);

    for i = 1:m
        t_gmres = @timed begin
            e1 = zeros(i+1,1);
            e1[1] = 1;
            Q, H = arnoldi(A,b,i)
            b_norm = norm(b)
            z = H\(b_norm .*e1);
            x = Q[:,1:i]*z;
        end
        if i == 1
            time_v[i] = t_gmres.time;
        else
            time_v[i] = time_v[i-1] + t_gmres.time;
        end
        res = norm(A*x-b)/norm(b);
        res_v[i] = norm(A*x-b)/norm(b);	          # residual norm
        err_v[i] = norm(x-x_exact)/norm(x_exact); # error norm
        if res/res_v[1] < tol
            break
        end
    end
    return x, res_v, err_v, time_v
end
