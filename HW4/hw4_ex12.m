com_mode = input('Enter mode:\n');
nv = [10:20:100];
tol = 1e-10;

switch com_mode
    case 1
        alpha = 0;
        fprintf('Backslash on vectorized system, alpha = %6.4f\n', alpha)
        t_1 = zeros(length(nv),1);
        err_1 = zeros(length(nv),1);
        for i = 1:length(nv)
            n = nv(i);
            [Dxx, G_un, F] = assemble_sysmatrices(n);
            I = eye(n);
            Dv = kron(Dxx,I) + kron(I,Dxx);
            g_un = diag(G_un(:));            
            A = Dv + alpha*g_un;
            fv = F(:);
            tStart = tic;
            x1 = A\fv;
            t_1(i) = toc(tStart);
            X1 = reshape(x1,n,length(x1)/n);
            err_1(i) = norm(Dxx*X1+X1*Dxx-F);
        end        
    case 2
        alpha = 0;
        fprintf('Lyapunov solver, alpha = %6.4f\n', alpha)
        t_2 = zeros(length(nv),1);
        err_2 = zeros(length(nv),1);
        for i = 1:length(nv)
            n = nv(i);
            [Dxx, G_un, F] = assemble_sysmatrices(n);
            tStart = tic;
            X2 = lyap(Dxx,-F);
            t_2(i) = toc(tStart);
            err_2(i) = norm(Dxx*X2+X2*Dxx-F);
        end         
    case 3
        alpha = 1;
        fprintf('Backslash on vectorized system, alpha = %6.4f\n', alpha)
        t_3 = zeros(length(nv),1);
        err_3 = zeros(length(nv),1);
        for i = 1:length(nv)
            n = nv(i);
            [Dxx, G_un, F] = assemble_sysmatrices(n);
            I = eye(n);
            Dv = kron(Dxx,I) + kron(I,Dxx);
            g_un = diag(G_un(:));            
            A = Dv + alpha*g_un;
            fv = F(:);
            tStart = tic;
            x3 = A\fv;
            t_3(i) = toc(tStart);
            X3 = reshape(x3,n,length(x3)/n);
            err_3(i) = norm(Dxx*X3+X3*Dxx+alpha*G_un.*X3-F);
        end
    case 4
        alpha = 1;
        fprintf('GMRES on vectorized system, alpha = %6.4f\n', alpha)
        t_4 = zeros(length(nv),1);
        err_4 = zeros(length(nv),1);
        for i = 1:length(nv)
            n = nv(i);
            [Dxx, G_un, F] = assemble_sysmatrices(n);
            I = eye(n);
            Dv = kron(Dxx,I) + kron(I,Dxx);
            g_un = diag(G_un(:));            
            A = Dv + alpha*g_un;
            fv = F(:);
            tStart = tic;
            x4 = gmres(A,fv,[],tol,100);
            t_4(i) = toc(tStart);
            X4 = reshape(x4,n,length(x4)/n);
            err_4(i) = norm(Dxx*X4+X4*Dxx+alpha*G_un.*X4-F);
        end
    case 5
        alpha = 1;
        fprintf('Preconditioned GMRES on vectorized system, alpha = %6.4f\n', alpha)
        t_5 = zeros(length(nv),1);
        err_5 = zeros(length(nv),1);
        for i = 1:length(nv)
            n = nv(i);
            [Dxx, G_un, F] = assemble_sysmatrices(n);
            I = eye(n);
            Dv = kron(Dxx,I) + kron(I,Dxx);
            g_un = diag(G_un(:));            
            A = Dv + alpha*g_un;
            fv = F(:);
            tStart = tic;
            x5 = gmres(A,fv,[],tol,[],Dv);
            t_5(i) = toc(tStart);
            X5 = reshape(x5,n,length(x5)/n);
            err_5(i) = norm(Dxx*X5+X5*Dxx+alpha*G_un.*X5-F);
        end
    case 6
        alpha = 0.1;
        fprintf('Preconditioned GMRES on vectorized system, alpha, alpha = %6.4f\n', alpha)
        t_6 = zeros(length(nv),1);
        err_6 = zeros(length(nv),1);
        for i = 1:length(nv)
            n = nv(i);
            [Dxx, G_un, F] = assemble_sysmatrices(n);
            I = eye(n);
            Dv = kron(Dxx,I) + kron(I,Dxx);
            g_un = diag(G_un(:));            
            A = Dv + alpha*g_un;
            fv = F(:);
            tStart = tic;
            x6 = gmres(A,fv,[],tol,[],Dv);
            t_6(i) = toc(tStart);
            X6 = reshape(x6,n,length(x6)/n);
            err_6(i) = norm(Dxx*X6+X6*Dxx+alpha*G_un.*X6-F);
        end
    otherwise
        error('Enter a number between 1 and 6\n')
end