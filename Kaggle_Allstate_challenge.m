%% Neural networks (NN) were trained to predict allstate policy purchasing.
%Policy A,C,D,F,G predictions were performed using NN pattern recognition
%toolbox implemented in matlab.
%Policy B, E were predicted with a customized NN model

%% Train and test data loading and preparation
load train_Clean.txt;
load test_Clean.txt;

%Transpose cleaned train and test data
X=transpose(train_Clean);
Xtest=transpose(test_Clean);
%Load 7 policies as targets, transpose them and remove the original lable.
%Only keep the columns that indicating classes
load yA.txt;
yA=transpose(yA);
yA=yA(2:end,:);
load yB.txt;
yB=transpose(yB);
yB=yB(2:end,:);
load yC.txt;
yC=transpose(yC);
yC=yC(2:end,:);
load yD.txt;
yD=transpose(yD);
yD=yD(2:end,:);
load yE.txt;
yE=transpose(yE);
yE=yE(2:end,:);
load yF.txt;
yF=transpose(yF);
yF=yF(2:end,:);
load yG.txt;
yG=transpose(yG);
yG=yG(2:end,:);

%% Train NN as a classifier to predict each policy using models implemented in
%ClassPred.m
[testIndicesA,yAtest]=ClassPred(X,yA,Xtest,13);
[testIndicesB,yBtest]=ClassPred(X,yB,Xtest,12);
[testIndicesC,yCtest]=ClassPred(X,yC,Xtest,13);
[testIndicesD,yDtest]=ClassPred(X,yD,Xtest,13);
[testIndicesE,yEtest]=ClassPred(X,yE,Xtest,12);
[testIndicesF,yFtest]=ClassPred(X,yF,Xtest,13);
[testIndicesG,yGtest]=ClassPred(X,yG,Xtest,13);

%% Export predictions to .txt files for further formating in R
outputA=transpose(testIndicesA);
save predictionA.txt outputA -ASCII;
outputB=transpose(testIndicesB);
save predictionB.txt outputB -ASCII;
outputC=transpose(testIndicesC);
save predictionC.txt outputC -ASCII;
outputD=transpose(testIndicesD);
save predictionD.txt outputD -ASCII;
outputE=transpose(testIndicesE);
save predictionE.txt outputE -ASCII;
outputF=transpose(testIndicesF);
save predictionF.txt outputF -ASCII;
outputG=transpose(testIndicesG);
save predictionG.txt outputG -ASCII;












