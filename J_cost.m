% Calculate the loss function without the non-differentiable part
function f_x = J_cost(W, X, F, G0, alpha, r, ...
                      lambda0, lambda1, lambda2, lambda4)
    numView = length(X);
    kmeans_c = 0;
    balance_c = 0;
    for v = 1: numView
        p = size(X{v}, 1);
        E{v} = ((X{v} - F{v}*G0').*sqrt(ones(p, 1)*(W.*W)'))';
        Ei2{v} = sum(sum(E{v}.*E{v}, 2));
        kmeans_c = kmeans_c + (alpha(v)^r)*sum(Ei2{v});
        balance_c = balance_c + (alpha(v)^r)*sum(balance_cost(W, X{v}'));
    end

    f_x = lambda0*kmeans_c...
         +lambda1*balance_c...
         +lambda2*((W.*W)'*(W.*W))...         
         +lambda4*(sum(W.*W)-1)^2;
end
