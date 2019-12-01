clc
close all
clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             Data Preparation                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T = xlsread('breast-cancer-wisconsin.csv'); % Read data
T(:,1) = []; % Delete first column with IDs

C = T(:,10); % Copy class labels to another matrix
T(:,10) = []; % Delete last column with class labels
C(C==2)=0; % Transform benign class label (2) to 0
C(C==4)=1; % Transform malignant class label (4) to 1

%Normalize data to [0,1]
normT = T - min(T(:));
normT = normT ./ max(normT(:));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    K-Fold Cross Validation (k=10) with Discriminant Analysis (linear)   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Create indices for the 10-fold cross-validation.
indices = crossvalind('Kfold',C,10);

%Initialize an object to measure the performance of the classifier.
cp = classperf(C);

% Perform the classification using the measurement data and report the error rate, 
% which is the ratio of the number of incorrectly classified samples divided by the
% total number of classified samples.
for i = 1:10
    test = (indices == i); 
    train = ~test;
    class = classify(normT(test,:),normT(train,:),C(train,:),'linear');
    classperf(cp,class,test);
end
fprintf('K-Fold Cross Validation (k=10) with Discriminant Analysis (linear)\n')
fprintf('Accuracy: %f\n',1- cp.ErrorRate);
fprintf('Sensitivity: %f\n',cp.Sensitivity);
fprintf('Specificity: %f\n\n',cp.Specificity);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K-Fold Cross Validation (k=10) with Discriminant Analysis (mahalanobis) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Create indices for the 10-fold cross-validation.
indices = crossvalind('Kfold',C,10);

%Initialize an object to measure the performance of the classifier.
cp = classperf(C);

% Perform the classification using the measurement data and report the error rate, 
% which is the ratio of the number of incorrectly classified samples divided by the
% total number of classified samples.
for i = 1:10
    test = (indices == i); 
    train = ~test;
    class = classify(normT(test,:),normT(train,:),C(train,:),'mahalanobis');
    classperf(cp,class,test);
end
fprintf('K-Fold Cross Validation (k=10) with Discriminant Analysis (mahalanobis)\n')
fprintf('Accuracy: %f\n',1- cp.ErrorRate);
fprintf('Sensitivity: %f\n',cp.Sensitivity);
fprintf('Specificity: %f\n\n',cp.Specificity);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K-Fold Cross Validation (k=10) with K-Nearest Neighbor (NumNeighbors=5) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Create indices for the 10-fold cross-validation.
indices = crossvalind('Kfold',C,10);

%Initialize an object to measure the performance of the classifier.
cp = classperf(C);

% Perform the classification using the measurement data and report the error rate, 
% which is the ratio of the number of incorrectly classified samples divided by the
% total number of classified samples.
for i = 1:10
    test = (indices == i); 
    train = ~test;
    mdl = fitcknn(normT(train,:),C(train,:),'NumNeighbors',5);
    class = predict(mdl,normT(test,:));
    classperf(cp,class,test);
end
fprintf('K-Fold Cross Validation (k=10) with K-Nearest Neighbor (NumNeighbors=5)\n')
fprintf('Accuracy: %f\n',1- cp.ErrorRate);
fprintf('Sensitivity: %f\n',cp.Sensitivity);
fprintf('Specificity: %f\n\n',cp.Specificity);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K-Fold Cross Validation (k=10) with K-Nearest Neighbor (NumNeighbors=25) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Create indices for the 10-fold cross-validation.
indices = crossvalind('Kfold',C,10);

%Initialize an object to measure the performance of the classifier.
cp = classperf(C);

% Perform the classification using the measurement data and report the error rate, 
% which is the ratio of the number of incorrectly classified samples divided by the
% total number of classified samples.
for i = 1:10
    test = (indices == i); 
    train = ~test;
    mdl = fitcknn(normT(train,:),C(train,:),'NumNeighbors',25);
    class = predict(mdl,normT(test,:));
    classperf(cp,class,test);
end
fprintf('K-Fold Cross Validation (k=10) with K-Nearest Neighbor (NumNeighbors=25)\n')
fprintf('Accuracy: %f\n',1- cp.ErrorRate);
fprintf('Sensitivity: %f\n',cp.Sensitivity);
fprintf('Specificity: %f\n\n',cp.Specificity);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K-Fold Cross Validation (k=10) with Naive Bayes (Gaussian distribution) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Create indices for the 10-fold cross-validation.
indices = crossvalind('Kfold',C,10);

%Initialize an object to measure the performance of the classifier.
cp = classperf(C);

% Perform the classification using the measurement data and report the error rate, 
% which is the ratio of the number of incorrectly classified samples divided by the
% total number of classified samples.
for i = 1:10
    test = (indices == i); 
    train = ~test;
    mdl = fitcnb(normT(train,:),C(train,:),'DistributionNames','normal');
    class = predict(mdl,normT(test,:));
    classperf(cp,class,test);
end
fprintf('K-Fold Cross Validation (k=10) with Naive Bayes (Gaussian distribution)\n')
fprintf('Accuracy: %f\n',1- cp.ErrorRate);
fprintf('Sensitivity: %f\n',cp.Sensitivity);
fprintf('Specificity: %f\n\n',cp.Specificity);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  K-Fold Cross Validation (k=10) with Naive Bayes (Kernel distribution)  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Create indices for the 10-fold cross-validation.
indices = crossvalind('Kfold',C,10);

%Initialize an object to measure the performance of the classifier.
cp = classperf(C);

% Perform the classification using the measurement data and report the error rate, 
% which is the ratio of the number of incorrectly classified samples divided by the
% total number of classified samples.
for i = 1:10
    test = (indices == i); 
    train = ~test;
    mdl = fitcnb(normT(train,:),C(train,:),'DistributionNames','kernel');
    class = predict(mdl,normT(test,:));
    classperf(cp,class,test);
end
fprintf('K-Fold Cross Validation (k=10) with Naive Bayes (Kernel distribution)\n')
fprintf('Accuracy: %f\n',1- cp.ErrorRate);
fprintf('Sensitivity: %f\n',cp.Sensitivity);
fprintf('Specificity: %f\n\n',cp.Specificity);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K-Fold Cross Validation (k=10) with Support Vector Machines (BoxConstraint=1) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Create indices for the 10-fold cross-validation.
indices = crossvalind('Kfold',C,10);

%Initialize an object to measure the performance of the classifier.
cp = classperf(C);

% Perform the classification using the measurement data and report the error rate, 
% which is the ratio of the number of incorrectly classified samples divided by the
% total number of classified samples.
for i = 1:10
    test = (indices == i); 
    train = ~test;
    mdl = fitcsvm(normT(train,:),C(train,:),'BoxConstraint',1);
    class = predict(mdl,normT(test,:));
    classperf(cp,class,test);
end
fprintf('K-Fold Cross Validation (k=10) with Support Vector Machines (BoxConstraint=1)\n')
fprintf('Accuracy: %f\n',1- cp.ErrorRate);
fprintf('Sensitivity: %f\n',cp.Sensitivity);
fprintf('Specificity: %f\n\n',cp.Specificity);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K-Fold Cross Validation (k=10) with Support Vector Machines (BoxConstraint=10) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Create indices for the 10-fold cross-validation.
indices = crossvalind('Kfold',C,10);

%Initialize an object to measure the performance of the classifier.
cp = classperf(C);

% Perform the classification using the measurement data and report the error rate, 
% which is the ratio of the number of incorrectly classified samples divided by the
% total number of classified samples.
for i = 1:10
    test = (indices == i); 
    train = ~test;
    mdl = fitcsvm(normT(train,:),C(train,:),'BoxConstraint',10);
    class = predict(mdl,normT(test,:));
    classperf(cp,class,test);
end
fprintf('K-Fold Cross Validation (k=10) with Support Vector Machines (BoxConstraint=10)\n')
fprintf('Accuracy: %f\n',1- cp.ErrorRate);
fprintf('Sensitivity: %f\n',cp.Sensitivity);
fprintf('Specificity: %f\n\n',cp.Specificity);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K-Fold Cross Validation with Decision Tree (AlgorithmForCategorical=Exact) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            Exact algorithm for best categorical predictor split:           % 
%                                                                            %
%                  "Consider all 2^(C-1) - 1 combinations"                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Create indices for the 10-fold cross-validation.
indices = crossvalind('Kfold',C,10);

%Initialize an object to measure the performance of the classifier.
cp = classperf(C);

% Perform the classification using the measurement data and report the error rate, 
% which is the ratio of the number of incorrectly classified samples divided by the
% total number of classified samples.
for i = 1:10
    test = (indices == i); 
    train = ~test;
    mdl = fitctree(normT(train,:),C(train,:),'AlgorithmForCategorical','Exact');
    class = predict(mdl,normT(test,:));
    classperf(cp,class,test);
end
fprintf('K-Fold Cross Validation (k=10) with Decision Tree (AlgorithmForCategorical=Exact)\n')
fprintf('Accuracy: %f\n',1- cp.ErrorRate);
fprintf('Sensitivity: %f\n',cp.Sensitivity);
fprintf('Specificity: %f\n\n',cp.Specificity);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    K-Fold Cross Validation with Decision Tree (AlgorithmForCategorical=PCA)   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             PCA algorithm for best categorical predictor split:               % 
%                                                                               %
% "Compute a score for each category using the inner product between the first  %
%  principal component of a weighted covariance matrix (of the centered class   %
%  probability matrix) and the vector of class probabilities for that category. % 
%  Sort the scores in ascending order, and consider all C - 1 splits."          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Create indices for the 10-fold cross-validation.
indices = crossvalind('Kfold',C,10);

%Initialize an object to measure the performance of the classifier.
cp = classperf(C);

% Perform the classification using the measurement data and report the error rate, 
% which is the ratio of the number of incorrectly classified samples divided by the
% total number of classified samples.
for i = 1:10
    test = (indices == i); 
    train = ~test;
    mdl = fitctree(normT(train,:),C(train,:),'AlgorithmForCategorical','PCA');
    class = predict(mdl,normT(test,:));
    classperf(cp,class,test);
end
fprintf('K-Fold Cross Validation (k=10) with Decision Tree (AlgorithmForCategorical=PCA)\n')
fprintf('Accuracy: %f\n',1- cp.ErrorRate);
fprintf('Sensitivity: %f\n',cp.Sensitivity);
fprintf('Specificity: %f\n\n',cp.Specificity);
