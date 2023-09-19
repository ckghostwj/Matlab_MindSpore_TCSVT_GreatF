function [Con_P,s_we,obj] = GreatF(X,Z_graph,ind_folds,G,numClust,lambda1,lambda2,lambda3,max_iter)

for iv = 1:length(X)
    ind_11{iv} = find(ind_folds(:,iv)==1);
end

numInst = size(ind_folds,1);
for iv = 1:length(X)
    rand('twister',757*iv);
%    rand('seed',757*iv);
   U{iv} = orth(rand(size(X{iv},1),numClust)); 
   Piv{iv} = U{iv}'*X{iv};
   s_we{iv} = ones(size(X{iv},1),1);
end

for iter = 1:max_iter
    % -------- Q (Con_P)------ %
    linshi_PZG = 0;
    linshi_GDG = 0;
    for iv = 1:length(X)
        linshi_PZG = linshi_PZG+(Piv{iv}*Z_graph{iv}*G{iv}');
        diag_D = sum(Z_graph{iv},1);
        linshi_D = zeros(numInst,1);
        linshi_D(ind_11{iv},1) = diag_D;
        linshi_GDG = linshi_GDG+linshi_D;
    end
    Con_P = linshi_PZG*diag(1./max(linshi_GDG,eps));
    % ----------- Piv -------- %
    for iv = 1:length(X)
        linshi_D = diag(sum(Z_graph{iv},1));
        lyap_A = U{iv}'*diag(s_we{iv})*U{iv};
        lyap_B = lambda3*linshi_D;
        lyap_C = U{iv}'*diag(s_we{iv})*X{iv} + lambda3*Con_P*G{iv}*Z_graph{iv}';
        Piv{iv} = lyap(lyap_A,lyap_B,-lyap_C);
    end
    % -------- U{iv} -------- %
    for iv = 1:length(X)
        U{iv} = X{iv}*Piv{iv}'/(Piv{iv}*Piv{iv}'+lambda2*eye(size(Piv{iv},1)));
    end
    % ------------ s_we ---- %
    for iv = 1:length(X)
        linshi_E = X{iv}-U{iv}*Piv{iv};
        linshi_h = diag(linshi_E*linshi_E')+lambda2*diag(U{iv}*U{iv}');
        linshi_s = -0.5/lambda1*linshi_h;
        s_we{iv} = EProjSimplex_new(linshi_s);
    end
    % --------- obj -------- %
    Rec_error = 0;
     for iv = 1:length(X)
        linshi_P = Con_P*G{iv};           
        graph_D = diag(sum(Z_graph{iv}));
        Rec_error = Rec_error+trace(Piv{iv}*graph_D*Piv{iv}')+trace(linshi_P*graph_D*linshi_P')-2*trace(Piv{iv}*Z_graph{iv}*linshi_P');
     end   
    linshi_obj = 0;
    for iv = 1:length(X)
        linshi_obj = linshi_obj + norm(diag(sqrt(s_we{iv}))*(X{iv}-U{iv}*Piv{iv}),'fro')^2+lambda1*sum(s_we{iv}.^2)+lambda2*norm(diag(sqrt(s_we{iv}))*U{iv},'fro')^2;
    end
    obj(iter) = linshi_obj+lambda3*Rec_error';
    if iter > 3 && abs(obj(iter)-obj(iter-1))<1e-4
        iter
        break;
    end    
end
end