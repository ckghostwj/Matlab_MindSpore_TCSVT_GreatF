% This is the code of paper: Jie Wen, et al. Graph Regularized and Feature Aware Matrix Factorization for Robust Incomplete Multi-view Clustering, TCSVT, 2023.
% For any problems, please contact: jiewen_pr@126.com

clear all
clc

Dataname = 'bbcsport4vbigRnSp';

% ------- para ---- %

percentDel = 0.5;
para_k = 5;
lambda1 = 1000000;
lambda2 = 1000;
lambda3 = 0.000001;
lambda4 = 0.1;

f = 3;
Datafold = [Dataname,'_percentDel_',num2str(percentDel),'.mat'];
load(Dataname);         % 一列一个样本
load(Datafold);
[numFold,numInst] = size(folds);
numClust = length(unique(truth));
numInst  = length(truth); 
load(Dataname);
ind_folds = folds{f};            
truthF = truth;

if size(X{1},2)~=length(truth) || size(X{2},2)~=length(truth)
    for iv = 1:length(X)
        X{iv} = X{iv}';
    end
end
linshi_AAW = 0;
linshi_WWW= 0;
S_ini = 0;
clear Y
for iv = 1:length(X)
    X1 = X{iv};
    X1 = NormalizeFea(X1,0);
    ind_0 = find(ind_folds(:,iv) == 0);
    X1(:,ind_0) = [];
    Y{iv} = X1;

    linshi_G = diag(ind_folds(:,iv));
    linshi_G(:,ind_0) = [];  % n*nv
    G{iv} = linshi_G;

    options = [];
    options.NeighborMode = 'KNN';
    options.k = para_k;
    options.WeightMode = 'HeatKernel';      % Binary  HeatKernel
    linshi_W = full(constructW(X1',options));
    linshi_W = (linshi_W+linshi_W')*0.5;
    Z_graph{iv} = linshi_W*lambda4 + eye(size(X1,2));
end
clear X
X = Y;
clear Y;

max_iter = 100;

% ------------ main code -------------- %
[Con_P,s_we,obj] = GreatF(X,Z_graph,ind_folds,G,numClust,lambda1,lambda2,lambda3,max_iter);

new_F = Con_P';
norm_mat = repmat(sqrt(sum(new_F.*new_F,2)),1,size(new_F,2));
for i = 1:size(norm_mat,1)
    if (norm_mat(i,1)==0)
        norm_mat(i,:) = 1;
    end
end
new_F = new_F./norm_mat; 
%     rand('seed',7578);
pre_labels    = kmeans(new_F,numClust,'emptyaction','singleton','replicates',20,'display','off');
result_cluster = ClusteringMeasure(truthF, pre_labels)*100

