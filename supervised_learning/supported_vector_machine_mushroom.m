%% clean the workspace and command line
function supported_vector_machine_mushroom

clear;
clc;

%% read data
filename = 'mushroom\agaricus-lepiota.data';
fin = fopen(filename, 'r');
format = ['%s %s %s %s %s %s %s %s %s %s' ...
          '%s %s %s %s %s %s %s %s %s %s' ...
                       '%s %s %s'];
rawdata = textscan(fin,format,'delimiter',',');
fclose(fin);

%% shuffle the data
rawdata = shuffle(rawdata);

%% convert to normalized numeric and k-fold partition
Y = rawdata{1};
xdata = [rawdata(2:11), rawdata(13:23)];
cv = cvpartition(Y,'k',10);

%% run
figure; hold on;
for i = 1:cv.NumTestSets

fprintf(1,'Training %dth fold ...\n',i);
    
X_training = String2Matrix(xdata,cv.training(i));
Y_training = Y(cv.training(i));

X_test = String2Matrix(xdata,cv.test(i));
Y_test = Y(cv.test(i));

% progressively use more data for training
N = 20;
size_block = cv.TrainSize(i)/N;
trainig_size = [];
training_err_rate_noprune = [];
test_err_rate_noprune = [];

trainig_size = [1, 3, 6, 10:20:190, 200:size_block:cv.TrainSize(i)];
%trainig_size = [1, 3, 6, 10:30:190, 200:200:2000];

for j = 1:length(trainig_size)
    
   trainig_size(j) = min(floor(trainig_size(j)),cv.TrainSize(i));

% create a svm template
t = templateSVM('KernelFunction','linear', ...
               'KernelOffset',0,'KernelScale',1, 'Verbose',2);

% devide multi-classification problem into multiple binary classification
% problmes. For each binary learner, one class is positive and the rest are
% negative. This design exhausts all combinations of positive class assignments.
svm = fitcecoc(X_training(1:trainig_size(j),:),Y_training(1:trainig_size(j)), ...
                  'Coding','onevsall','Learner',t,'Verbose', 2);
              
%testing
Y_training_prediction = predict(svm,X_training);
Y_test_prediction = predict(svm,X_test);
training_err_rate(j) = sum(~strcmp(Y_training_prediction,Y_training))/length(Y_training);
test_err_rate(j) = sum(~strcmp(Y_test_prediction,Y_test))/length(Y_test);

end

if i == 1 % initialize size
mean_training_err_rate = 0*training_err_rate;
mean_test_err_rate = 0*test_err_rate;
end

mean_training_err_rate = (mean_training_err_rate*(i-1)+training_err_rate)/i;
mean_test_err_rate = (mean_test_err_rate*(i-1)+test_err_rate)/i;
%plot(trainig_size, training_err_rate,'b-');
%plot(trainig_size, test_err_rate,'r-');
end

p(1) = plot(trainig_size, mean_training_err_rate,'b-o');
p(2) = plot(trainig_size, mean_test_err_rate,'r-o');

xlabel('Training Size',  'Fontsize', 16);
ylabel('Percentage Error',  'Fontsize', 16);
set(gcf, 'color', 'white');
set(gca, 'box', 'on', 'Fontsize', 14, 'linewidth', 1);
set(p,  'linewidth', 1.5, 'markersize', 8);
le = legend('Training Set', 'Test Set');
%set(le, 'box', 'off');
text(mean(trainig_size)/2, 0.3, 0, 'Support Vector Machine - Mushroom Data','Fontsize', 16);
save;

return;


%% convert matrix array to normalized numeric matrix
function matrix = String2Matrix(stringMatrix,id)
    matrix = [];
    for i = 1:length(stringMatrix)  
      matrix = [matrix, String2Num(stringMatrix{i},id)];
    end
return


%% convert string array to normalized numeric array
function numArray = String2Num(stringArray,id)
    stringArray=stringArray(id);
    str = getStruct(stringArray);
    numArray =[];
    for i = 1:length(stringArray)
        if ~isempty(regexp(stringArray{i}(1),'[0-9]', 'once'))
           fieldname = ['number',stringArray{i}];
        else  
           fieldname = stringArray{i};
        end
        numArray(i,1) =  str.(fieldname);
    end
return


%% helper structure for converting string to numeric
 function str = getStruct(stringArray)
 str = struct();
 % assign numerical values
 n = 0;
 for i=1:length(stringArray)
     if ~isempty(regexp(stringArray{i}(1),'[0-9]', 'once'))
         fieldname = ['number',stringArray{i}];
     else
         fieldname = stringArray{i};
     end
     if ~isfield(str,fieldname)
         str.(fieldname)= n;
         n = n + 1;
     end
 end
 % normalize to [0,1]
 fields = fieldnames(str); 
 for i= 1:numel(fields)
     str.(fields{i}) = str.(fields{i})/max((n-1),1);
 end
return;


function newdata = shuffle(data)
      
    N = length(data); % # of collumn
    K = length(data{1}); % # of samples
    
    id = randperm(K); % random permutation
    newdata = {};
  for n = 1:N
      col = {};
    for i = 1:length(id)
       col = [col; data{n}(id(i))];
    end
     newdata = [newdata, {col}];
  end
  
return;


