filename = 'diabetes.txt'; urlwrite('http://www.stanford.edu/~hastie/Papers/LARS/diabetes.data',filename);
formatSpec = '%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', '\t', 'HeaderLines' ,1, 'ReturnOnError', false);
fclose(fileID);
diabetes = table(dataArray{1:end-1}, 'VariableNames', {'AGE','SEX','BMI','BP','S1','S2','S3','S4','S5','S6','Y'});
clearvars filename delimiter startRow formatSpec fileID dataArray ans;

% Delete the file
delete diabetes.txt

predNames = diabetes.Properties.VariableNames(1:end-1);
X = diabetes{:,1:end-1};
y = diabetes{:,end};

% Test for linear independence
A = [1,1,1;1,2,3;4,4,4]
A = transpose(A)

