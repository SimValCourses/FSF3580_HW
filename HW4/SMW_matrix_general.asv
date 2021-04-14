function [X] = SMW_matrix_general(U1,V1,U2,V2,U3,V3,U4,V4,A,F)
% function [X] = SMW_matrix_general(U1,V1,U2,V2,U3,V3,U4,V4,A,F)
% Solve AX + XA' + M1 X M2' + M3 X M4' = F,
% where M_i=U_i*V_i' and U_i, V_i are low rank s_i,
% via the Sherman-Morrison-Woodbury matrix-oriented formula
%

symmA=(norm(A-A',1)<1e-12);
nonmulti=isempty(U3);

n=size(A,1);
s1=size(U1,2); s2=size(U2,2);
s3=size(U3,2); s4=size(U4,2);

if (symmA)
    [Q,R]=eig(A); L=diag(R)*ones(1,n)+ones(n,1)*diag(R).';
    rhs=(Q'*F*Q)./L;
else
    [Q,R]=schur(A,'real');
    rhs=lyap(R,-Q'*F*Q);
end

p=0;

% Change of basis
U1Q=Q'*U1; U2Q=Q'*U2;
V1Q=Q'*V1; V2Q=Q'*V2;

if (~nonmulti) % extra low-rank term
    U3Q=Q'*U3; U4Q=Q'*U4;
    V3Q=Q'*V3; V4Q=Q'*V4;
end

% computed blocks of H associated with M1,M2
for k=1:s2
    if (symmA), G=U2Q(1:n,k)'./L;end % replica of the row U2Q((:,k)'
    for i=1:s1
        p=p+1;
        if (symmA)
            VV1(1:n,1:n,p)=U1Q(1:n,i).*G; % replica of column U1Q(:,i)
        else
            VV1(1:n,1:n,p)=lyap(R,-U1Q(:,i)*U2Q(:,k)');
        end

        vk=V1Q'*(VV1(1:n,1:n,p)*V2Q);
        H(1:s1*s2,p)=reshape(vk,s1*s2,1);
        if (~nonmulti)
            vk=V3Q'*(VV1(1:n,1:n,p)*V4Q);
            H(s1*s2+1:s1*s2+s3*s4,p)=reshape(vk,s3*s4,1);
        end
    end
end
coef(1:s1*s2,1)=reshape(V1Q'*rhs*V2Q,s1*s2,1);

% computed blocks of H associated with M3,M4
% for k=1:s4
%     if (symmA), G=U4Q(1:n,k)'./L;end % replica of the row u4Q((:,k)'
%     for i=1:s3
%         p=p+1;
%         if (symmA)
%             VV1(1:n,1:n,p)=U3Q(1:n,i).*G; % replica of column U3Q(:,i)
%         else
%             VV1(1:n,1:n,p)=lyap(R,-U3Q(:,i)*U4Q(:,k)');
%         end
% 
%         vk=V1Q'*(VV1(1:n,1:n,p)*V2Q);
%         H(1:s1*s2,p)=reshape(vk,s1*s2,1);
%         vk=V3Q'*(VV1(1:n,1:n,p)*V4Q);
%         H(s1*s2+1:s1*s2+s3*s4,p)=reshape(vk,s3*s4,1);
%     end
% end
if (~nonmulti)
    coef(s1*s2+1:s1*s2+s3*s4,1)=reshape(V3Q'*rhs*V4Q,s3*s4,1);
end

C=(eye(s1*s2+s3*s4)+H)\coef;
Stot=reshape(VV1,n^2,s1*s2+s3*s4)*C;
X=Q*(rhs-reshape(Stot,n,n))*Q';