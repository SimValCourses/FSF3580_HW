nv = [20; 60; 100; 200];
time1 = zeros(length(nv),1);
time2 = zeros(length(nv),1);
error_1 = zeros(length(nv),1);
error_2 = zeros(length(nv),1);
error_s = zeros(length(nv),1);
err_1r = zeros(length(nv),1);
err_2r = zeros(length(nv),1);

for i = 1:length(nv)
    n = nv(i);
    h = 1/(n+1);
    x = h*[1:n];
    y = linspace(h,1-h,n);
    [XX, YY] = meshgrid(x,y);

    G_un = zeros(n,n);
    G_un(n/4,n/2) = 1/h;
    F = abs(XX-YY);
    Dxx = second_der(n,h);
    mult_k = 1/h;

    % naive approach 
    I = eye(n);
    Dv = kron(Dxx,I) + kron(I,Dxx);
    g_un = diag(G_un(:));            
    A = Dv + g_un;
    f = F(:);
    tStart1 = tic;
    x1 = A\f;
    time1(i) = toc(tStart1); 
    X1 = reshape(x1,n,length(x1)/n);

    % Lyapunov approach - see Hao & Simoncini (2020)
    % solve equation AX + XA' + MXN = F
    U1 = zeros(n,1);
    U1(n/4) = 1/h;
    V1 = zeros(n,1);
    V1(n/4) = 1;
    U2 = zeros(n,1);
    U2(n/2) = 1;
    V2 = zeros(n,1);
    V2(n/2) = 1;
    U3 = [];
    V3 = [];
    U4 = [];
    V4 = [];
    tStart2 = tic;
    X2 = SMW_matrix_general(U1,V1,U2,V2,U3,V3,U4,V4,full(Dxx),F);
    time2(i) = toc(tStart2); 

    % check the correctness of the methods
    error_1(i) = norm(Dxx*X1+X1*Dxx+G_un.*X1-F)/norm(F);
    error_2(i) = norm(Dxx*X2+X2*Dxx+G_un.*X2-F)/norm(F);
    error_s(i) = norm(X2-X1);

    % Check algorithm
    X_ref = randn(n);
    F_ref = Dxx*X_ref+X_ref*Dxx+G_un.*X_ref;
    x1r = A\F_ref(:);
    X1r = reshape(x1r,n,length(x1r)/n);
    X2r = SMW_matrix_general(U1,V1,U2,V2,U3,V3,U4,V4,full(Dxx),F_ref);
    err_1r(i) = norm(X1r-X_ref)/norm(X_ref);
    err_2r(i) = norm(X2r-X_ref)/norm(X_ref);
end
