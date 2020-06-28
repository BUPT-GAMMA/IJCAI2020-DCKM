function [outX_train] = data_filter(inX_train)
N = size(inX_train,1);
M = size(inX_train,2);

n = 0;
for mt = 1:M
    if abs(sum(inX_train(:,mt))) == 0
        n = n + 1;
    end
    if abs(sum(inX_train(:,mt))) == N
        n = n + 1;
    end
end

fprintf('nnnnnnnnnnnn.......... %d...\n', n);
outX_train = zeros(N, M-n);
i = 1;
for m = 1:M
    if abs(sum(inX_train(:,m))) ~= 0 
        if abs(sum(inX_train(:,m))) ~= N
            outX_train(:, i) = inX_train(:,m);
            i = i + 1;
        end
    end
end
end
