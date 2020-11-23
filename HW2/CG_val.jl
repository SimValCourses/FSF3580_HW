function my_cg_val(A,b,m, x_exact)
    r = copy(b);
    p = copy(b);
    x = zeros(length(b),1);
    res_v = zeros(m,1);
    err_v = zeros(m,1);
    time_v = zeros(m,1);

    for i = 1:m
        t_cg = @timed begin
            alpha = (r'*r)/(p'*A*p);
            x = x + alpha .*p;
            r_old = r;
            r = r_old - alpha .*(A*p);
            beta = (r'*r)/(r_old'*r_old);
            p = r + beta .*p;
        end
        if i == 1
            time_v[i] = t_cg.time;
        else
            time_v[i] = time_v[i-1] + t_cg.time;
        end
        res_v[i]= norm(A*x-b)/norm(b)	         # residual norm
        err_v[i]= norm(x-x_exact)/norm(x_exact); # error norm
    end
    return x, res_v, err_v, time_v
end
