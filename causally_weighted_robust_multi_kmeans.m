function [ outG0, outW, outFCell, outAlpha, outAlpha_r, outObj, outNumIter ] = causally_weighted_robust_multi_kmeans( inXCell, inPara, inG0, lambda0, lambda1, lambda2, lambda3, lambda4)
% solve the following problem
% min_{F^(v), G0, alpha^{v}} sum_v {(alpha^{v})^r*||X^(v) - F^(v)G0^T)^T||_2,1}
% s.t. G0 is a cluster indicator, sum_v{alpha^{v}) = 1, alpha^{v} >= 0
% 
% input: 
%       inXcell: v by 1 cell, and the size of each cell is d_v by n
%       inPara: parameter cell
%               inPara.maxIter: max number of iterator
%               inPara.thresh:  the convergence threshold
%               inPara.numCluster: the number cluster
%               inPara.r: the parameter to control the distribution of the
%                         weights for each view
%       inG0: init common cluster indicator
% output:
%       outG0: the output cluster indicator (n by c)
%       outFcell: the cluster centroid for each view (d by c by v)
%       outObj: obj value
%       outNumIter: number of iterator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ref:
% Xiao Cai, Feiping Nie, Heng Huang. 
% Multi-View K-Means Clustering on Big Data. 
% The 23rd International Joint Conference on Artificial Intelligence (IJCAI), 2013.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% parameter settings
maxIter = inPara.maxIter;
thresh = inPara.thresh;
c = inPara.numCluster;
r = inPara.r;
n = size(inXCell{1}, 2);
numView = length(inXCell);
W = ones(n, 1);
W_prev = W;
parameter_iter = 0.5;

lambda_W = 1;
W_All = zeros(n, maxIter);
% inti alpha
alpha = ones(numView, 1)/numView; 
% inti common indicator D3
G0 = inG0;
% % Fix D3{v}, G0, alpha, update F{v}
% for v = 1: numView
%     M = G0'*D4{v}*G0;
%     N = inXCell{v}*D4{v}*G0;
%     F{v} = N/M;
% end
% clear M N;
tmp = 1/(1-r);
obj = zeros(maxIter, 1);
% loop
for t = 1: maxIter
    fprintf('processing iteration %d...\n', t);
    
    % Fix D3{v}, G0, alpha, update F{v}
    for v = 1: numView
        p = size(inXCell{v}, 1);
        M = ((G0'.*(ones(c, 1)*(W.*W)'))*G0);
        N = (inXCell{v}.*(ones(p, 1)*(W.*W)'))*G0;
        F{v} = N/M;
    end
    fprintf('update F done...\n');
        
    % Fix D3{v}, F{v}, update G0    
    for i = 1:n
        for v = 1: numView
            xVec{v} = inXCell{v}(:,i);
        end
        G0(i,:) = searchBestIndicator(alpha, r, xVec, F);
    end

    fprintf('update G0 done...\n');
    % Fix G, alpha{v}, F{v} update W
    % Update W
    y = W;
    W = W+(t/(t+3))*(W-W_prev);    
    f_base = J_cost(W, inXCell, F, G0, alpha, r, lambda0, lambda1, lambda2, lambda4);

    obj1 = zeros(n, 1);
    obj2 = zeros(n, 1);
    for v = 1: numView
        p = size(inXCell{v}, 1);
        obj1 = obj1 + (alpha(v)^r) * ((ones(1,p)*((inXCell{v}-F{v}*G0').*(inXCell{v} - F{v}*G0')))'.*W);
        obj2 = obj2 + (alpha(v)^r) * balance_grad(W, inXCell{v}')*ones(p, 1);
    end
    
    grad_W = lambda0*obj1...
            +lambda1*obj2...
            +4*lambda2*W.*W.*W...           
            +4*lambda4*(sum(W.*W)-1)*W;
   % W = W-lambda_W*grad_W;
   % fprintf('grad_W');
   % grad_W
    while 1
        z = prox_l1(W-lambda_W*grad_W, 0);
        if J_cost(z, inXCell, F, G0, alpha, r, lambda0, lambda1, lambda2, lambda4)...
                <= f_base + grad_W'*(z-W) ...
                + (1/(2*lambda_W))*sum((z-W).^2)
                        break;
        end 
        lambda_W = parameter_iter*lambda_W;
    end
    W_prev = y;
    W = z;    
    W_All(:,t) = W;
    
      
    fprintf('update W done...\n');
    
    % Fix F{v}, G0, D4{v}, update alpha
    h = zeros(numView, 1);
    b = zeros(numView, 1);
    for v = 1: numView
        p = size(inXCell{v}, 1);
        E{v} = ((inXCell{v} - F{v}*G0').*sqrt(ones(p, 1)*(W.*W)'))';
        Ei2{v} = sum(E{v}.*E{v}, 2);
        h(v) = sum(Ei2{v});
        b(v) = sum(balance_cost(W, inXCell{v}'));
    end
    alpha = ((r*(h+lambda3*b)).^tmp)/(sum(((r*(h+lambda3*b)).^tmp)));
    
    fprintf('update alpha done...\n');
    
    obj(t) = J_cost(W, inXCell, F, G0, alpha, r, ....
                          lambda0, lambda1, lambda2, lambda4);
             
    if t > 1 && abs(obj(t) - obj(t-1)) < thresh || t == maxIter
        break
    end
    % calculate the obj
end
% debug
% figure, plot(1: length(obj), obj);

outW = W.*W;
outObj = obj;
outNumIter = t;
outFCell = F;
outG0 = G0;
outAlpha = alpha;
outAlpha_r = alpha.^r;

end
%% function searchBestIndicator
function outVec = searchBestIndicator(alpha, r, xCell, F)
% solve the following problem,
numView = length(F);
c = size(F{1}, 2);
tmp = eye(c);
obj = zeros(c, 1);
for j = 1: c
    for v = 1: numView
        obj(j,1) = obj(j,1) + (alpha(v)^r) * (norm(xCell{v} - F{v}(:,j))^2);
    end
end
[min_val, min_idx] = min(obj);
outVec = tmp(:, min_idx);
end

