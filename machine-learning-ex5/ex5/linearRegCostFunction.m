function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = (X*theta);
linear_reg = sum((h - y) .^ 2);

theta_ = theta;
theta_(1) = 0;
reg = (lambda / (2*m)) * sum(theta_ .^ 2);

J = linear_reg / (2*m)  + reg;


% calculate gradient
grad(1) = sum((h - y) * X(1)) / m;
grad(2:end) = (X' * (h .- y) / m)(2:end) + lambda/m * theta(2:end);

% =========================================================================

grad = grad(:);

end
