function X=lyap0(A,P);
%  Solves the Lyapunov equation
%    AX+XA'+P=0
%  with the Bartel-Stewart algorithm
%  The result is equivalent to lyap(A,P)
%
%  Example:
%     A=randn(5); P=randn(5);
%     X0=lyap0(A,P);
%     should_be_zero=norm(A*X0+X0*A'+P)
% 
%  Exercise in Numerical linear algebra, KTH, 2015 


% For compatibility with the matlab command lyap 
W=-P;

%% Step 1
% Computes a real Schur form and 
% determine a vector nv with 1 or 2 
% specifying the size of that block
[Q,T]=schur(A);
mode=1;
nv=[1];
for k=1:length(A)-1
    if (T(k+1,k)~=0)
        nv(end)=nv(end)+1;
    else
        nv(end+1)=1;
    end
end
n=length(A);
CC=Q'*W*Q; %% Step 2

m=n;       %% Step 3
Y=zeros(n,n);
r=length(nv);
for k=r:-1:1;
    m=m-nv(k);   %% Step 5
    nk=nv(k);    

    %% Step 6:
    C11=CC(1:m,1:m);
    C12=CC(1:m,m+(1:nk));
    C21=CC(m+(1:nk),1:m);
    C22=CC(m+(1:nk),m+(1:nk));
    
    %% Step 7
    R11=T(1:m,1:m);
    R12=T(1:m,m+(1:nk));
    R22=T(m+(1:nk),m+(1:nk));

    %% Step 8:
    Z22=...
        
    %% Step 9: compute \tilde{C}_{12} and \tilde{C}_{21}
    C12t=...
    C21t=...
    
    %% Step 10
    % more efficient way to compute the following
    % by backward subsitution
    % z12=(kron(R22,eye(m))+kron(eye(nk),R11))\reshape(C12t,m*nk,1);
    % Z12=reshape(z12,m,nk);
    % z21T=(kron(R22,eye(m))+kron(eye(nk),R11))\reshape(C21t,m*nk,1);
    % Z21=reshape(z21T,m,nk)';
    %
    Z12=backsolve(C12t,T,R22,nv(1:k-1));
    Z21=backsolve(C21t,T,R22,nv(1:k-1))';

    
    
    Y(1:m,m+(1:nk))=Z12;        %% Step 11
    Y(m+(1:nk),1:m)=Z21;        %% Step 12
    Y(m+(1:nk),m+(1:nk))=Z22;   %% Step 13
    %% Step 14
    C11=C11-R12*Z21-Z12*R12';   
    CC=C11;                     
    
end

%% Step 16
X=Q*Y*Q';



function X=backsolve(D,T,R22,nv)
% Solves a triangular Lyapunov equation
% using Lemma 5.2.1

X=zeros(size(D));
nr=size(D,2);
R22XEYE=kron(R22,eye(2));
for j=length(nv):-1:1
    nj=nv(j);
    mj=sum(nv(1:(j-1)));

    %% Compute Wjt with matrix vector multiplys
    m0=sum(nv(1:j-1));
    QQ=T(mj+(1:nj),m0+1:sum(nv(1:end)))* X(m0+1:sum(nv(1:end)),:);
    Wjt=D(mj+(1:nj),1:nr)-QQ;


    Tjj=T(mj+(1:nj),mj+(1:nj));
    if (nr==1 && nj==1)
        Xj=Wjt/(Tjj+R22);
    else        
        wtj=reshape(Wjt,nj*nr,1);
        if (nr>1)
            if (nj>1)
                AA=kron(eye(nr),Tjj)+R22XEYE;                     
            else % nj=1
                AA=eye(nr)*Tjj+R22;                
            end
        else
            AA=Tjj+R22*eye(nj);
        end
        xj=AA(1:(nj*nr),1:(nj*nr))\wtj;
        Xj=reshape(xj,nj,nr);
    end

    %should_be_zero=norm(Tjj*Xj+Xj*R22'-Wjt)
    X(mj+(1:nj),:)=Xj;
end