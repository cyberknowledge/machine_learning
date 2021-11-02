%%% Outine %%%
% 1. Prepare dataset
% 2. Divide dataset into training and testing sets
% 3. Preparing validation set of training set (K-fold CV)
% 4. Feature selection
% 5. Finding best parameters
% 6. Test the model with test set
% 7. Visualize hyperplane

% Clean everything
clear; close all; clc;

%% Step 1 - Preparing dataset

load fisheriris

% Change label strings into numbers
species_num = grp2idx(species);

% SVM is a binary classifier
% Binary classification

% Just as an excersize, let's create some useless features
X = randn(100,10);

% Insert real data into columns 1,3,5,7
X(:,[1,3,5,7]) = meas(1:100,:); % 1, 3, 5, 7
y = species_num(1:100);

%% Step 2 - Training and Testing Sets
% Create a string that randomly permites integers from 1 to 100
rand_num = randperm(size(X,1));

% Create a train set with 80 samples
X_train = X(rand_num(1:round(0.8*length(rand_num))),:);
y_train = y(rand_num(1:round(0.8*length(rand_num))),:);

% Create a test set with 20 samples
X_test = X(rand_num(round(0.2*length(rand_num))+1:end),:);
y_test = y(rand_num(round(0.2*length(rand_num))+1:end),:);

%% Step 3 - Make use of CV partition to create validation set of training set

c = cvpartition(y_train,'k',5);

%% Step 4 - Feature Selection
% sequentialfs(fun,X_train, y_train,'cv',c,'options',opts,'nfeatures',2)
opts = statset('display','iter');
classf = @(train_data, train_labels, test_data, test_labels)...
    sum(predict(fitcsvm(train_data, train_labels,'KernelFunction','rbf'), test_data) ~= test_labels);

[fs, history] = sequentialfs(classf, X_train, y_train, 'cv', c, 'options', opts,'nfeatures',2);

%% Step 5 - Best hyperparameter

% Get the best features that are useful for the classification
X_train_w_best_feature = X_train(:,fs);

Md1 = fitcsvm(X_train_w_best_feature,y_train,'KernelFunction','rbf','OptimizeHyperparameters','auto',...
      'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
      'expected-improvement-plus','ShowPlots',true)); % Bayes' Optimization


%% Step 6 - Final test with test set
X_test_w_best_feature = X_test(:,fs);
test_accuracy_for_iter = sum((predict(Md1,X_test_w_best_feature) == y_test))/length(y_test)*100

%% Step 7 - Hyperplane

figure;
hgscatter = gscatter(X_train_w_best_feature(:,1),X_train_w_best_feature(:,2),y_train);
hold on;
h_sv=plot(Md1.SupportVectors(:,1),Md1.SupportVectors(:,2),'ko','markersize',8);


% test set

gscatter(X_test_w_best_feature(:,1),X_test_w_best_feature(:,2),y_test,'rb','xx')

% decision plane
XLIMs = get(gca,'xlim');
YLIMs = get(gca,'ylim');
[xi,yi] = meshgrid([XLIMs(1):0.01:XLIMs(2)],[YLIMs(1):0.01:YLIMs(2)]);
dd = [xi(:), yi(:)];
pred_mesh = predict(Md1, dd);
redcolor = [1, 0.8, 0.8];
bluecolor = [0.8, 0.8, 1];
pos = find(pred_mesh == 1);
h1 = plot(dd(pos,1), dd(pos,2),'s','color',redcolor,'Markersize',5,'MarkerEdgeColor',redcolor,'MarkerFaceColor',redcolor);
pos = find(pred_mesh == 2);
h2 = plot(dd(pos,1), dd(pos,2),'s','color',bluecolor,'Markersize',5,'MarkerEdgeColor',bluecolor,'MarkerFaceColor',bluecolor);
uistack(h1,'bottom');
uistack(h2,'bottom');
legend([hgscatter;h_sv],{'setosa','versicolor','support vectors'})
