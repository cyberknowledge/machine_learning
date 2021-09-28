% Set Random Number Generator
rng(2017)
x_lr = (linspace(1,15,100))';
y_lr = 2*x_lr + (x_lr+rand(size(x_lr))).^2;

% Plot the data
scatter(x_lr,y_lr)
xlabel('x')
ylabel('y')

% B has only one parameter as we only have a single column in x_lr with 100
% values
X_lr = [x_lr];

% Solve for Bols
% Go back to Lesson 3 - Slide 27
%%% Bols = dot(x_lr,y_lr)/dot(x_lr,x_lr)
%%% Bols2 = sum(pinv(x_lr)*y_lr)

Bols_1 = X_lr\y_lr
Bols_2 = (X_lr'*X_lr)\(X_lr'*y_lr)
Bols_3 = inv(X_lr'*X_lr)*(X_lr'*y_lr)
Bols_4 = pinv(X_lr'*X_lr)*(X_lr'*y_lr)

hold on
plot(x_lr, X_lr*Bols_1)
title('y = \Bols X','FontSize',18)
hold off
