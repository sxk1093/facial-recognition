


function [W,D] = pcadimred(X)
    
    x = double(X);
    % subtract mean
    mean_matrix = mean(x, 2);
    x =  x - mean_matrix;
    
    % calculate covariance
    s = cov(x');
    
    % obtain eigenvalue & eigenvector
    [V, D] = eig(s);
    eigval = diag(D);
    
    % sort eigenvalues in descending order
    eigval = eigval(end: - 1:1);
    V = fliplr(V);

    W = V;
    D = eigval;
end