function d = dist(x, y) 

    % x - bigger matrix
    % y - smaller matrix

    [M, N] = size(x); 
    [M2, P] = size(y); 
    
    if (M ~= M2) 
        error('Matrix dimensions do not match.') 
    end 
    
    d = zeros(N, P);
    
    for i=1:P 
        % Euclidian Distance
        d(:, i) = sum((repmat(y(:,i), [1, N])-x).^2).^0.5; 
    end
end 